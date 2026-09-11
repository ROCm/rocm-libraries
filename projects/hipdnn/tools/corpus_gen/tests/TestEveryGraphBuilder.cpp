// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestEveryGraphBuilder.cpp
 * @brief Drives every registered builder through the registry, not just the shipped ones.
 *
 * The gap this closes: every other registry test iterates the *shipped declarations*, and only
 * seven operations have one. Fifteen of the twenty-two adapters were therefore registered,
 * reachable by name, and never once executed by the suite -- so their argument names, required
 * argument lists, enum resolution and tensor-uid assignment were unverified. That is exactly
 * where a role spelled `invRms` in the adapter and `inv_rms` in a future declaration hides,
 * and it would surface much later as "this operation mysteriously builds nothing".
 *
 * Each case below is a synthetic declaration naming one builder. The assertions are the ones
 * that catch a wrong adapter rather than a wrong graph: the build succeeds, the bytes are a
 * readable Graph, it has the tensor count that builder writes, the declared element type
 * reaches the header, and every tensor carries a distinct uid.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace hipdnn_corpus_gen
{
namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

using Ints = std::vector<int64_t>;

/// A constant vector argument -- dims, strides, padding, a window.
nlohmann::json ints(const std::string& name, const Ints& values)
{
    return {{"name", name}, {"kind", "constant"}, {"constant", values}};
}

/// A constant scalar argument of any JSON type.
nlohmann::json scalar(const std::string& name, const nlohmann::json& value)
{
    return {{"name", name}, {"kind", "constant"}, {"constant", value}};
}

/// `<role>Dims` and `<role>Strides` for one tensor role, as the tensorRole helper expects.
void addRole(nlohmann::json& arguments, const std::string& role)
{
    arguments.push_back(ints(role + "Dims", Ints{2, 4}));
    arguments.push_back(ints(role + "Strides", Ints{4, 1}));
}

/// One builder, the arguments it needs, and how many tensors it should write.
struct BuilderCase
{
    std::string function;
    std::vector<std::string> roles;   ///< tensor roles taking <role>Dims/<role>Strides
    std::vector<nlohmann::json> extra; ///< everything else the adapter reads
    size_t tensors = 0;

    /// Tensors this builder types independently of the operands, by name.
    ///
    /// Named per builder rather than globally: exempting "scale" everywhere would have quietly
    /// stopped checking layernorm's scale, which *is* the declared type. The exemptions are
    /// deliberate design decisions -- a normalization epsilon is an fp32 scalar whatever the
    /// activations are, an MoE token offset is an index, and a block-scale factor has its own
    /// declared scaleDataType, which is the entire point of quantization.
    std::set<std::string> foreignTypes;
};

nlohmann::json declarationFor(const BuilderCase& builder)
{
    nlohmann::json arguments = nlohmann::json::array();
    for(const auto& role : builder.roles)
    {
        addRole(arguments, role);
    }
    for(const auto& argument : builder.extra)
    {
        arguments.push_back(argument);
    }
    // Declared, not defaulted: an adapter that ignored it would otherwise pass by accident.
    arguments.push_back(scalar("dataType", "half"));

    return nlohmann::json{
        {"schema_version", "1.0"},
        {"operation", builder.function},
        {"parameters", {{"placeholder", {{"type", "int64"}}}}},
        {"stratification_axis", "working_set"},
        {"regimes", nlohmann::json::object()},
        {"graph_builder",
         {{"function", builder.function},
          {"source", "hipdnn_corpus_gen/GraphBuilders.hpp"},
          {"arguments", arguments}}}};
}

/// Geometry shared by the three convolution directions and the two resample ones.
std::vector<nlohmann::json> convGeometry()
{
    return {ints("prePadding", {0, 0}),
            ints("postPadding", {0, 0}),
            ints("convStrides", {1, 1}),
            ints("convDilation", {1, 1})};
}

std::vector<nlohmann::json> resampleGeometry()
{
    return {ints("window", {2, 2}),
            ints("poolStrides", {2, 2}),
            ints("prePadding", {0, 0}),
            ints("postPadding", {0, 0})};
}

std::vector<nlohmann::json> withGeometry(std::vector<nlohmann::json> head,
                                         const std::vector<nlohmann::json>& geometry)
{
    head.insert(head.end(), geometry.begin(), geometry.end());
    return head;
}

/// Every builder the registry knows, with the arguments its adapter actually reads.
///
/// Tensor counts are the builder's own addTensor calls, so a builder that silently stopped
/// writing an operand -- a batchnorm that lost its running statistics, say -- fails here rather
/// than producing a graph that is merely a different operation.
std::vector<BuilderCase> everyBuilder()
{
    return {
        {"convolutionForward", {"x", "w", "y"}, convGeometry(), 3, {}},
        {"convolutionBackwardData", {"dy", "w", "dx"}, convGeometry(), 3, {}},
        {"convolutionBackwardWeights", {"x", "dy", "dw"}, convGeometry(), 3, {}},
        {"matmul", {"a", "b", "c"}, {}, 3, {}},
        {"pointwiseBinary",
         {},
         {ints("dims", {2, 4}), ints("strides", {4, 1}), scalar("mode", "ADD")},
         3,
         {}},
        {"reduction",
         {"in", "out"},
         {scalar("mode", "ADD"), scalar("deterministic", false)},
         2,
         {}},
        {"layernormForward",
         {},
         {ints("dims", {2, 4}), ints("strides", {4, 1}), scalar("normalizedDimCount", 1)},
         5,
         {"epsilon"}},
        {"rmsNormForward", {}, {ints("dims", {2, 4}), ints("strides", {4, 1})}, 4, {"epsilon"}},
        {"sdpaForward", {"q", "k", "v", "o"}, {scalar("causalMask", true)}, 4, {}},
        {"sdpaBackward",
         {"q", "k", "v", "o", "dO", "stats", "dq", "dk", "dv"},
         {scalar("causalMask", true)},
         9,
         {}},
        {"layernormBackward",
         {"dy", "x", "scale", "dx", "dscale", "dbias"},
         {scalar("normalizedDimCount", 1)},
         6,
         {}},
        {"rmsNormBackward", {"dy", "x", "scale", "invRms", "dx", "dscale"}, {}, 6, {}},
        {"batchnormForwardTraining",
         {"x", "scale", "bias", "epsilon", "y", "mean", "invVariance"},
         {},
         7,
         {"epsilon"}},
        {"batchnormInference", {"x", "mean", "invVariance", "scale", "bias", "y"}, {}, 6, {}},
        {"batchnormBackward", {"dy", "x", "scale", "dx", "dscale", "dbias"}, {}, 6, {}},
        {"resampleForward",
         {"x", "y"},
         withGeometry({scalar("resampleMode", "MAXPOOL"), scalar("paddingMode", "ZERO_PAD")},
                      resampleGeometry()),
         2,
         {}},
        {"resampleBackward",
         {"dy", "dx"},
         withGeometry({scalar("resampleMode", "MAXPOOL"), scalar("paddingMode", "ZERO_PAD")},
                      resampleGeometry()),
         2,
         {}},
        {"blockScaleQuantize",
         {"x", "y", "scale"},
         {scalar("blockSize", 32), scalar("transpose", false), scalar("scaleDataType", "float")},
         3,
         {"scale"}},
        {"blockScaleDequantize",
         {"x", "scale", "y"},
         {ints("blockSize", {32}), scalar("negativeScale", false),
          scalar("scaleDataType", "float")},
         3,
         {"scale"}},
        {"moeGroupedMatmul",
         {"token", "weight", "firstTokenOffset", "output"},
         {scalar("moeMode", "NONE"), scalar("topK", 2), scalar("offsetDataType", "int32")},
         4,
         {"firstTokenOffset"}},
        {"moeGroupedMatmulBackward",
         {"dOutput", "token", "firstTokenOffset", "dWeight"},
         {scalar("offsetDataType", "int32")},
         4,
         {"firstTokenOffset"}},
    };
}

const fb::Graph* asGraph(const builders::GraphBytes& bytes)
{
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    return fb::VerifyGraphBuffer(verifier) ? fb::GetGraph(bytes.data()) : nullptr;
}

} // namespace

TEST(TestEveryGraphBuilder, EveryBuilderBuildsAReadableGraph)
{
    for(const auto& builder : everyBuilder())
    {
        const auto parsed = parseOperationMetadata(declarationFor(builder));
        ASSERT_TRUE(parsed.ok())
            << builder.function << ": " << (parsed.errors.empty() ? "" : parsed.errors.front());

        const auto built
            = buildGraphFor(*parsed.metadata, ProblemPoint{{"placeholder", int64_t{1}}});
        ASSERT_TRUE(built.ok()) << builder.function << ": " << built.error;

        // Verified, not merely non-empty: an adapter that assembled a malformed buffer would
        // otherwise pass here and fail much later inside the frontend.
        const auto* graph = asGraph(built.bytes);
        ASSERT_NE(graph, nullptr) << builder.function << " produced an unreadable buffer";
        ASSERT_NE(graph->tensors(), nullptr) << builder.function << " wrote no tensors";
        EXPECT_EQ(graph->tensors()->size(), builder.tensors)
            << builder.function << " wrote a different number of tensors than it declares";
        ASSERT_NE(graph->nodes(), nullptr);
        EXPECT_EQ(graph->nodes()->size(), 1U) << builder.function << " is not a single-node graph";
    }
}

TEST(TestEveryGraphBuilder, EveryBuilderGivesItsTensorsDistinctUids)
{
    // Two operands sharing a uid is a graph that looks right and aliases two buffers. The
    // nine-tensor SDPA backward adapter assigns uids by loop index, which is exactly the shape
    // of code where an off-by-one collides two of them.
    for(const auto& builder : everyBuilder())
    {
        const auto parsed = parseOperationMetadata(declarationFor(builder));
        ASSERT_TRUE(parsed.ok()) << builder.function;
        const auto built
            = buildGraphFor(*parsed.metadata, ProblemPoint{{"placeholder", int64_t{1}}});
        ASSERT_TRUE(built.ok()) << builder.function << ": " << built.error;

        const auto* graph = asGraph(built.bytes);
        ASSERT_NE(graph, nullptr);

        std::set<int64_t> uids;
        for(const auto* tensor : *graph->tensors())
        {
            EXPECT_TRUE(uids.insert(tensor->uid()).second)
                << builder.function << " reuses tensor uid " << tensor->uid();
        }
        EXPECT_EQ(uids.size(), builder.tensors) << builder.function;
    }
}

TEST(TestEveryGraphBuilder, EveryBuilderPropagatesTheDeclaredDataType)
{
    // A builder that hardcodes float would pass every structural check above while making the
    // dtype column of every training row it produces a lie.
    for(const auto& builder : everyBuilder())
    {
        const auto parsed = parseOperationMetadata(declarationFor(builder));
        ASSERT_TRUE(parsed.ok()) << builder.function;
        const auto built
            = buildGraphFor(*parsed.metadata, ProblemPoint{{"placeholder", int64_t{1}}});
        ASSERT_TRUE(built.ok()) << builder.function << ": " << built.error;

        const auto* graph = asGraph(built.bytes);
        ASSERT_NE(graph, nullptr);
        EXPECT_EQ(graph->io_data_type(), fb::DataType::HALF)
            << builder.function << " did not carry the declared dtype into the graph header";

        for(const auto* tensor : *graph->tensors())
        {
            const auto name = tensor->name() == nullptr ? std::string() : tensor->name()->str();
            if(builder.foreignTypes.count(name) == 1)
            {
                continue;
            }
            EXPECT_EQ(tensor->data_type(), fb::DataType::HALF)
                << builder.function << " tensor '" << name << "' has the wrong element type";
        }
    }
}

TEST(TestEveryGraphBuilder, EveryRegisteredBuilderIsCoveredByThisFile)
{
    // Without this, adding a builder and forgetting to add a case leaves it in exactly the
    // untested state the file exists to end -- and the suite would still be green.
    std::set<std::string> covered;
    for(const auto& builder : everyBuilder())
    {
        covered.insert(builder.function);
    }

    for(const auto& name : registeredBuilders())
    {
        // customOperation is deliberately unregistered-by-design elsewhere; if it ever gains a
        // registration it needs a case here too.
        EXPECT_EQ(covered.count(name), 1U) << name << " is registered but has no case here";
    }
}

TEST(TestEveryGraphBuilder, AnAdapterMissingAnArgumentSaysWhichOne)
{
    // The refusal path, which is how a mistyped declaration is meant to be diagnosed. If the
    // message did not name the argument, a nine-tensor operation would report only that
    // something was missing.
    for(const auto& builder : everyBuilder())
    {
        if(builder.roles.empty())
        {
            continue;
        }
        auto declaration = declarationFor(builder);
        auto& arguments = declaration["graph_builder"]["arguments"];

        const auto dropped = builder.roles.front() + "Dims";
        for(auto it = arguments.begin(); it != arguments.end(); ++it)
        {
            if((*it)["name"] == dropped)
            {
                arguments.erase(it);
                break;
            }
        }

        const auto parsed = parseOperationMetadata(declaration);
        ASSERT_TRUE(parsed.ok()) << builder.function;
        const auto built
            = buildGraphFor(*parsed.metadata, ProblemPoint{{"placeholder", int64_t{1}}});

        EXPECT_FALSE(built.ok()) << builder.function << " built a graph without " << dropped;
        EXPECT_FALSE(built.error.empty()) << builder.function << " refused without saying why";
    }
}

} // namespace hipdnn_corpus_gen

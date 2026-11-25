// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestEveryGraphBuilder.cpp
 * @brief Drives every registered builder through the registry, not just the shipped ones.
 *
 * Most builders have no shipped declaration, so each case is a synthetic one naming a single
 * builder. Checks: it builds, the buffer verifies, tensor and node counts match, the declared
 * dtype reaches the header, and tensor uids are distinct.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/node_operands_generated.h>

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <string_view>
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
    std::vector<std::string> roles; ///< tensor roles taking <role>Dims/<role>Strides
    std::vector<nlohmann::json> extra; ///< everything else the adapter reads
    size_t tensors = 0;

    /// Tensors this builder types independently of the operands, by name.
    ///
    /// Per builder, not global: exempting "scale" everywhere would skip layernorm's scale,
    /// which does take the declared type.
    std::set<std::string> foreignTypes;

    /// Nodes the graph holds: one for every builder but a fusion.
    size_t nodes = 1;
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
    // Non-default, so an adapter that ignores it fails.
    arguments.push_back(scalar("dataType", "half"));

    return nlohmann::json{{"schema_version", "1.0"},
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
/// Tensor counts are the builder's own addTensor calls, so a dropped operand fails here.
std::vector<BuilderCase> everyBuilder()
{
    return {
        {"convolutionForward", {"x", "w", "y"}, convGeometry(), 3, {}},
        {"convolutionBackwardData", {"dy", "w", "dx"}, convGeometry(), 3, {}},
        {"convolutionBackwardWeights", {"x", "dy", "dw"}, convGeometry(), 3, {}},
        {"convolutionBiasActivation",
         {"x", "w", "bias", "y"},
         [] {
             auto extra = convGeometry();
             extra.push_back(scalar("mode", "RELU_FWD"));
             return extra;
         }(),
         6,
         {"conv_out", "bias_out"},
         3},
        {"matmul", {"a", "b", "c"}, {}, 3, {}},
        {"matmulEpilogue",
         {"a", "b", "c", "bias"},
         {scalar("epilogue", "bias_activation"), scalar("mode", "RELU_FWD")},
         6,
         {"matmul_out", "bias_out"},
         3},
        {"blockScaledMatmul",
         {"a", "aScale", "b", "bScale", "c"},
         {scalar("scaleDataType", "fp8_e8m0"),
          scalar("outputDataType", "fp16"),
          nlohmann::json{{"name", "blockSize"}, {"kind", "constant"}, {"constant", 32}}},
         7,
         {"aScale", "bScale", "c", "a_dequantized", "b_dequantized"},
         3},
        {"pointwiseBinary",
         {},
         {ints("dims", {2, 4}), ints("strides", {4, 1}), scalar("mode", "ADD")},
         3,
         {}},
        {"pointwiseUnary",
         {},
         {ints("dims", {2, 4}), ints("strides", {4, 1}), scalar("mode", "RELU_FWD")},
         2,
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
        {"layernormForward",
         {},
         {ints("dims", {2, 4}),
          ints("strides", {4, 1}),
          scalar("normalizedDimCount", 1),
          scalar("phase", "training")},
         7,
         {"epsilon", "mean", "invVariance"}},
        {"rmsNormForward",
         {},
         {ints("dims", {2, 4}), ints("strides", {4, 1}), scalar("phase", "training")},
         5,
         {"epsilon", "invRms"}},
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
        {"layernormBackward",
         {"dy", "x", "scale", "dx", "dscale", "dbias", "mean", "invVariance"},
         {scalar("normalizedDimCount", 1)},
         8,
         {}},
        {"batchnormForwardTraining",
         {"x", "scale", "bias", "epsilon", "y", "mean", "invVariance"},
         {},
         7,
         {"epsilon"}},
        {"batchnormInference", {"x", "mean", "invVariance", "scale", "bias", "y"}, {}, 6, {}},
        {"batchnormInferenceActivationBackward",
         {"x", "mean", "invVariance", "scale", "bias", "dy", "dx", "dscale", "dbias"},
         {scalar("mode", "RELU_BWD")},
         11,
         {"bn_out", "dy_bn"},
         3},
        {"batchnormBackward", {"dy", "x", "scale", "dx", "dscale", "dbias"}, {}, 6, {}},
        {"batchnormBackward",
         {"dy", "x", "scale", "dx", "dscale", "dbias", "mean", "invVariance"},
         {},
         8,
         {}},
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
        ASSERT_TRUE(parsed.ok()) << builder.function << ": "
                                 << (parsed.errors.empty() ? "" : parsed.errors.front());

        const auto built
            = buildGraphFor(*parsed.metadata, ProblemPoint{{"placeholder", int64_t{1}}});
        ASSERT_TRUE(built.ok()) << builder.function << ": " << built.error;

        // Verified, not merely non-empty.
        const auto* graph = asGraph(built.bytes);
        ASSERT_NE(graph, nullptr) << builder.function << " produced an unreadable buffer";
        ASSERT_NE(graph->tensors(), nullptr) << builder.function << " wrote no tensors";
        EXPECT_EQ(graph->tensors()->size(), builder.tensors)
            << builder.function << " wrote a different number of tensors than it declares";
        ASSERT_NE(graph->nodes(), nullptr);
        EXPECT_EQ(graph->nodes()->size(), builder.nodes)
            << builder.function << " built the wrong number of nodes";
    }
}

TEST(TestEveryGraphBuilder, EveryBuilderGivesItsTensorsDistinctUids)
{
    // Shared uids would alias two buffers.
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
    // A builder that hardcodes float would mislabel every row's dtype.
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
    std::set<std::string> covered;
    for(const auto& builder : everyBuilder())
    {
        covered.insert(builder.function);
    }

    for(const auto& name : registeredBuilders())
    {
        EXPECT_EQ(covered.count(name), 1U) << name << " is registered but has no case here";
    }
}

TEST(TestEveryGraphBuilder, AnAdapterMissingAnArgumentSaysWhichOne)
{
    // This is how a mistyped declaration is diagnosed.
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

namespace
{

/// One shape every form is built at: GQA, causal, fp16.
constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 4;
constexpr int64_t HEADS_KV = 2;
constexpr int64_t SEQLEN_Q = 16;
constexpr int64_t SEQLEN_K = 32;
constexpr int64_t HEAD_DIM = 64;

ProblemPoint formPoint(const OperationMetadata& metadata,
                       const std::string& form,
                       bool stats = true,
                       const std::string& dtype = "fp16")
{
    return test::withDefaultCategoricals(metadata,
                                         ProblemPoint{{"batch", BATCH},
                                                      {"heads", HEADS},
                                                      {"heads_kv", HEADS_KV},
                                                      {"seqlen_q", SEQLEN_Q},
                                                      {"seqlen_k", SEQLEN_K},
                                                      {"head_dim", HEAD_DIM},
                                                      {"is_causal", true},
                                                      {"generate_stats", stats},
                                                      {"dtype", dtype},
                                                      {"form", form}});
}

/// What an SDPA node binds, as the generated `node_operands::visit` reports it for either
/// direction: its tensor operands by role with their uids, and the scalar attributes it sets. A
/// bool is listed only when true (the visitor reports an unset one at its default).
struct Operands
{
    std::map<std::string, int64_t> tensors;
    std::set<std::string> scalars;

    void tensor(std::string_view role, int64_t uid, bool /*workDataDependent*/)
    {
        tensors[std::string(role)] = uid;
    }

    void scalar(std::string_view name, int64_t /*value*/)
    {
        scalars.emplace(name);
    }

    void scalar(std::string_view name, double /*value*/)
    {
        scalars.emplace(name);
    }

    void scalar(std::string_view name, bool value)
    {
        if(value)
        {
            scalars.emplace(name);
        }
    }

    std::set<std::string> roles() const
    {
        std::set<std::string> roles;
        for(const auto& entry : tensors)
        {
            roles.insert(entry.first);
        }
        return roles;
    }
};

Operands operandsOf(const fb::Graph& graph)
{
    Operands bound;
    const auto* node = graph.nodes()->Get(0);
    if(const auto* forward = node->attributes_as_SdpaAttributes())
    {
        fb::node_operands::visit(*forward, bound);
    }
    else if(const auto* backward = node->attributes_as_SdpaBackwardAttributes())
    {
        fb::node_operands::visit(*backward, bound);
    }
    return bound;
}

/// What a form changes about the plain graph's node, and how many tensors its graph holds
/// (statistics included for the forward; the backward's nine roles always). A direction that
/// does not build the form has 0 tensors. The backward-only roles and scalars are added on top of
/// the shared ones for the backward (its ragged form carries the lengths the forward's does not).
struct FormExpectation
{
    std::string form;
    std::set<std::string> addedRoles;
    std::set<std::string> addedScalars;
    std::set<std::string> removedScalars;
    size_t forwardTensors;
    size_t backwardTensors;
    std::set<std::string> backwardAddedRoles = {};
    std::set<std::string> backwardAddedScalars = {};
};

std::vector<FormExpectation> formExpectations()
{
    return {
        {"plain", {}, {}, {}, 5, 9},
        {"scale_unset", {}, {}, {"attn_scale_value"}, 5, 9},
        {"scale_tensor", {"scale"}, {}, {"attn_scale_value"}, 6, 10},
        {"dropout", {"seed", "offset"}, {"dropout_probability"}, {}, 7, 11},
        {"padded", {"seq_len_q", "seq_len_kv"}, {"padding_mask"}, {}, 7, 11},
        // The offsets go on the tensors, not the node; the backward also binds the lengths.
        {"ragged", {}, {}, {}, 7, 13, {"seq_len_q", "seq_len_kv"}, {"padding_mask"}},
        {"bias", {"attn_mask"}, {}, {}, 6, 10},
        {"sink", {"sink_token"}, {}, {}, 6, 0},
    };
}

const fb::TensorAttributes* tensorWithUid(const fb::Graph& graph, std::optional<int64_t> uid)
{
    if(!uid.has_value())
    {
        return nullptr;
    }
    for(const auto* tensor : *graph.tensors())
    {
        if(tensor->uid() == *uid)
        {
            return tensor;
        }
    }
    return nullptr;
}

std::vector<int64_t> dimsOf(const fb::TensorAttributes* tensor)
{
    return tensor == nullptr ? std::vector<int64_t>{}
                             : std::vector<int64_t>(tensor->dims()->begin(), tensor->dims()->end());
}

BuildResult buildForm(const OperationMetadata& metadata,
                      const std::string& form,
                      bool stats = true,
                      const std::string& dtype = "fp16")
{
    return buildGraphFor(metadata, formPoint(metadata, form, stats, dtype));
}

} // namespace

namespace
{

/// Builds every form @p operation declares and checks that its node binds exactly the plain
/// node's operands and scalars, changed as the form says, over @p baseRoles.
void expectEveryFormBindsExactlyItsOptionalFields(const std::string& operation,
                                                  const std::set<std::string>& baseRoles)
{
    const bool backward = operation == "sdpa_bwd";
    const auto metadata = test::shippedOperation(operation);
    const auto* form = metadata.find("form");
    ASSERT_NE(form, nullptr) << operation;
    const auto tensorsOf = [backward](const FormExpectation& expected) {
        return backward ? expected.backwardTensors : expected.forwardTensors;
    };

    // The declared forms are the ones this direction builds: a new one needs an expectation here.
    std::set<std::string> buildable;
    for(const auto& expected : formExpectations())
    {
        if(tensorsOf(expected) != 0)
        {
            buildable.insert(expected.form);
        }
    }
    EXPECT_EQ(std::set<std::string>(form->values.begin(), form->values.end()), buildable)
        << operation;

    const auto plainBuilt = buildForm(metadata, "plain");
    ASSERT_TRUE(plainBuilt.ok()) << plainBuilt.error;
    const auto plain = operandsOf(*asGraph(plainBuilt.bytes));
    EXPECT_EQ(plain.roles(), baseRoles) << operation;
    ASSERT_EQ(plain.scalars.count("attn_scale_value"), 1U) << operation;

    for(const auto& expected : formExpectations())
    {
        if(tensorsOf(expected) == 0)
        {
            continue;
        }
        const auto built = buildForm(metadata, expected.form);
        ASSERT_TRUE(built.ok()) << operation << " " << expected.form << ": " << built.error;
        const auto* graph = asGraph(built.bytes);
        ASSERT_NE(graph, nullptr) << expected.form;
        const auto bound = operandsOf(*graph);

        auto roles = baseRoles;
        roles.insert(expected.addedRoles.begin(), expected.addedRoles.end());
        auto scalars = plain.scalars;
        for(const auto& name : expected.removedScalars)
        {
            scalars.erase(name);
        }
        scalars.insert(expected.addedScalars.begin(), expected.addedScalars.end());
        if(backward)
        {
            roles.insert(expected.backwardAddedRoles.begin(), expected.backwardAddedRoles.end());
            scalars.insert(expected.backwardAddedScalars.begin(),
                           expected.backwardAddedScalars.end());
        }

        EXPECT_EQ(bound.roles(), roles) << operation << " " << expected.form;
        EXPECT_EQ(bound.scalars, scalars) << operation << " " << expected.form;
        EXPECT_EQ(graph->tensors()->size(), tensorsOf(expected))
            << operation << " " << expected.form;
    }
}

} // namespace

TEST(TestSdpaForms, EveryForwardFormBindsExactlyItsOptionalFields)
{
    expectEveryFormBindsExactlyItsOptionalFields("sdpa_fwd", {"q", "k", "v", "o", "stats"});
}

TEST(TestSdpaForms, EveryBackwardFormBindsExactlyItsOptionalFields)
{
    // The backward schema has no sink token.
    expectEveryFormBindsExactlyItsOptionalFields(
        "sdpa_bwd", {"q", "k", "v", "o", "do", "stats", "dq", "dk", "dv"});
}

TEST(TestSdpaForms, ThePlainFormIsTheGraphTheDeclarationAlwaysBuilt)
{
    // No optional tensor, the stated scale: byte for byte what the builder writes given only
    // the operands, as it did before forms existed, and for each of the types declared before
    // the fp8 ones (no descale, and O in the inputs' type).
    const auto metadata = test::shippedOperation("sdpa_fwd");
    const auto tensor
        = [](int64_t uid, const char* name, Ints dims, Ints strides, fb::DataType type) {
              builders::TensorSpec spec;
              spec.uid = uid;
              spec.name = name;
              spec.dims = std::move(dims);
              spec.strides = std::move(strides);
              spec.dataType = type;
              return spec;
          };
    for(const auto& [declared, type] :
        std::vector<std::pair<std::string, fb::DataType>>{{"fp16", fb::DataType::HALF},
                                                          {"bf16", fb::DataType::BFLOAT16},
                                                          {"fp32", fb::DataType::FLOAT}})
    {
        const auto q = tensor(1,
                              "q",
                              {BATCH, HEADS, SEQLEN_Q, HEAD_DIM},
                              {SEQLEN_Q * HEADS * HEAD_DIM, HEAD_DIM, HEADS * HEAD_DIM, 1},
                              type);
        const auto k = tensor(2,
                              "k",
                              {BATCH, HEADS_KV, SEQLEN_K, HEAD_DIM},
                              {SEQLEN_K * HEADS_KV * HEAD_DIM, HEAD_DIM, HEADS_KV * HEAD_DIM, 1},
                              type);
        auto v = k;
        v.uid = 3;
        v.name = "v";
        auto o = q;
        o.uid = 4;
        o.name = "o";
        const auto stats = tensor(5,
                                  "stats",
                                  {BATCH, HEADS, SEQLEN_Q, 1},
                                  {HEADS * SEQLEN_Q, SEQLEN_Q, 1, 1},
                                  fb::DataType::FLOAT);
        builders::SdpaOptions options;
        options.causalMask = true;
        options.generateStats = true;
        options.attnScale = 0.125F; // rsqrt(64)
        const builders::GraphTypes types{type, fb::DataType::FLOAT, fb::DataType::FLOAT};

        const auto built = buildForm(metadata, "plain", true, declared);
        ASSERT_TRUE(built.ok()) << declared << ": " << built.error;
        EXPECT_EQ(built.bytes, builders::sdpaForward(q, k, v, o, options, types, stats))
            << declared;
    }
}

TEST(TestSdpaForms, TheScaleTensorIsAScalarPassByValueWhoseDefaultIsTheStatedScale)
{
    const auto built = buildForm(test::shippedOperation("sdpa_fwd"), "scale_tensor");
    ASSERT_TRUE(built.ok()) << built.error;
    const auto* graph = asGraph(built.bytes);
    const auto* attributes = graph->nodes()->Get(0)->attributes_as_SdpaAttributes();
    const auto* scale = tensorWithUid(*graph, attributes->scale_tensor_uid());
    ASSERT_NE(scale, nullptr);

    EXPECT_EQ(dimsOf(scale), (Ints{1, 1, 1, 1}));
    EXPECT_EQ(scale->data_type(), fb::DataType::FLOAT);
    EXPECT_TRUE(scale->is_runtime_pass_by_value());
    ASSERT_EQ(scale->value_type(), fb::TensorValue::Float32Value);
    EXPECT_FLOAT_EQ(scale->value_as_Float32Value()->value(), 0.125F);
}

TEST(TestSdpaForms, DropoutCarriesItsRateAndInt64SeedAndOffsetTensors)
{
    const auto built = buildForm(test::shippedOperation("sdpa_fwd"), "dropout");
    ASSERT_TRUE(built.ok()) << built.error;
    const auto* graph = asGraph(built.bytes);
    const auto* attributes = graph->nodes()->Get(0)->attributes_as_SdpaAttributes();

    ASSERT_TRUE(attributes->dropout_probability().has_value());
    EXPECT_FLOAT_EQ(attributes->dropout_probability().value(), 0.1F);
    for(const auto uid : {attributes->seed_tensor_uid(), attributes->offset_tensor_uid()})
    {
        const auto* tensor = tensorWithUid(*graph, uid);
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(dimsOf(tensor), (Ints{1, 1, 1, 1}));
        EXPECT_EQ(tensor->data_type(), fb::DataType::INT64);
    }
    EXPECT_NE(attributes->seed_tensor_uid().value(), attributes->offset_tensor_uid().value());
}

TEST(TestSdpaForms, PaddedNamesBothLengthsAsInt32OnePerBatch)
{
    for(const auto& operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto built = buildForm(test::shippedOperation(operation), "padded");
        ASSERT_TRUE(built.ok()) << operation << ": " << built.error;
        const auto* graph = asGraph(built.bytes);
        const auto bound = operandsOf(*graph);
        for(const auto* role : {"seq_len_q", "seq_len_kv"})
        {
            ASSERT_EQ(bound.tensors.count(role), 1U) << operation << " " << role;
            const auto* tensor = tensorWithUid(*graph, bound.tensors.at(role));
            ASSERT_NE(tensor, nullptr) << operation;
            EXPECT_EQ(dimsOf(tensor), (Ints{BATCH, 1, 1, 1})) << operation;
            EXPECT_EQ(tensor->data_type(), fb::DataType::INT32) << operation;
        }
    }
}

TEST(TestSdpaForms, TheBiasTakesQsTypeAndOneScorePerQueryAndKey)
{
    for(const auto& operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto built = buildForm(test::shippedOperation(operation), "bias");
        ASSERT_TRUE(built.ok()) << operation << ": " << built.error;
        const auto* graph = asGraph(built.bytes);
        const auto bound = operandsOf(*graph);
        ASSERT_EQ(bound.tensors.count("attn_mask"), 1U) << operation;
        const auto* bias = tensorWithUid(*graph, bound.tensors.at("attn_mask"));
        ASSERT_NE(bias, nullptr) << operation;
        EXPECT_EQ(dimsOf(bias), (Ints{1, 1, SEQLEN_Q, SEQLEN_K})) << operation;
        EXPECT_EQ(bias->data_type(), fb::DataType::HALF) << operation;
    }
}

TEST(TestSdpaForms, TheSinkIsOneFloatLogitPerQueryHead)
{
    const auto built = buildForm(test::shippedOperation("sdpa_fwd"), "sink");
    ASSERT_TRUE(built.ok()) << built.error;
    const auto* graph = asGraph(built.bytes);
    const auto* sink = tensorWithUid(
        *graph, graph->nodes()->Get(0)->attributes_as_SdpaAttributes()->sink_token_tensor_uid());
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(dimsOf(sink), (Ints{1, HEADS, 1, 1}));
    EXPECT_EQ(sink->data_type(), fb::DataType::FLOAT);
}

TEST(TestSdpaForms, RaggedPacksQOAndStatsThroughOneOffsetAndKVThroughAnother)
{
    for(const bool stats : {true, false})
    {
        const auto built = buildForm(test::shippedOperation("sdpa_fwd"), "ragged", stats);
        ASSERT_TRUE(built.ok()) << built.error;
        const auto* graph = asGraph(built.bytes);
        const auto* attributes = graph->nodes()->Get(0)->attributes_as_SdpaAttributes();

        // Compact THD: no length tensors, no padding mask.
        EXPECT_FALSE(attributes->padding_mask());
        EXPECT_FALSE(attributes->seq_len_q_tensor_uid().has_value());
        EXPECT_FALSE(attributes->seq_len_kv_tensor_uid().has_value());

        std::map<std::string, const fb::TensorAttributes*> roles{
            {"q", tensorWithUid(*graph, attributes->q_tensor_uid())},
            {"k", tensorWithUid(*graph, attributes->k_tensor_uid())},
            {"v", tensorWithUid(*graph, attributes->v_tensor_uid())},
            {"o", tensorWithUid(*graph, attributes->o_tensor_uid())}};
        if(stats)
        {
            roles["stats"] = tensorWithUid(*graph, attributes->stats_tensor_uid());
        }
        for(const auto& [name, tensor] : roles)
        {
            ASSERT_NE(tensor, nullptr) << name;
            ASSERT_TRUE(tensor->ragged_offset_tensor_uid().has_value()) << name;
        }

        // One offset tensor for the query side, one for the key side.
        EXPECT_EQ(roles.at("q")->ragged_offset_tensor_uid(),
                  roles.at("o")->ragged_offset_tensor_uid());
        EXPECT_EQ(roles.at("k")->ragged_offset_tensor_uid(),
                  roles.at("v")->ragged_offset_tensor_uid());
        EXPECT_NE(roles.at("q")->ragged_offset_tensor_uid(),
                  roles.at("k")->ragged_offset_tensor_uid());
        for(const auto* name : {"q", "k"})
        {
            const auto* offsets = tensorWithUid(*graph, roles.at(name)->ragged_offset_tensor_uid());
            ASSERT_NE(offsets, nullptr) << name;
            EXPECT_EQ(dimsOf(offsets), (Ints{BATCH + 1, 1, 1, 1})) << name;
            EXPECT_EQ(offsets->data_type(), fb::DataType::INT32) << name;
        }

        // Elements per token: heads * head_dim for Q and O, heads_kv * head_dim for K and V,
        // heads for the statistics.
        EXPECT_EQ(roles.at("q")->ragged_offset_multiplier(), HEADS * HEAD_DIM);
        EXPECT_EQ(roles.at("o")->ragged_offset_multiplier(), HEADS * HEAD_DIM);
        EXPECT_EQ(roles.at("k")->ragged_offset_multiplier(), HEADS_KV * HEAD_DIM);
        EXPECT_EQ(roles.at("v")->ragged_offset_multiplier(), HEADS_KV * HEAD_DIM);
        if(stats)
        {
            EXPECT_EQ(roles.at("stats")->ragged_offset_multiplier(), HEADS);
        }
        EXPECT_EQ(graph->tensors()->size(), stats ? 7U : 6U);
    }
}

TEST(TestSdpaForms, BackwardRaggedPacksTheQuerySideThroughOneOffsetAndTheKeySideThroughAnother)
{
    // cuDNN's THD backward: Q, O, dO, dQ and the statistics through one offset tensor, K, V, dK
    // and dV through another, with the per-batch lengths and the padding mask beside them.
    const auto built = buildForm(test::shippedOperation("sdpa_bwd"), "ragged");
    ASSERT_TRUE(built.ok()) << built.error;
    const auto* graph = asGraph(built.bytes);
    const auto* attributes = graph->nodes()->Get(0)->attributes_as_SdpaBackwardAttributes();
    ASSERT_NE(attributes, nullptr);

    EXPECT_TRUE(attributes->padding_mask());
    for(const auto uid : {attributes->seq_len_q_tensor_uid(), attributes->seq_len_kv_tensor_uid()})
    {
        const auto* lengths = tensorWithUid(*graph, uid);
        ASSERT_NE(lengths, nullptr);
        EXPECT_EQ(dimsOf(lengths), (Ints{BATCH, 1, 1, 1}));
        EXPECT_EQ(lengths->data_type(), fb::DataType::INT32);
    }

    const std::map<std::string, const fb::TensorAttributes*> query{
        {"q", tensorWithUid(*graph, attributes->q_tensor_uid())},
        {"o", tensorWithUid(*graph, attributes->o_tensor_uid())},
        {"dO", tensorWithUid(*graph, attributes->do_tensor_uid())},
        {"dq", tensorWithUid(*graph, attributes->dq_tensor_uid())},
        {"stats", tensorWithUid(*graph, attributes->stats_tensor_uid())}};
    const std::map<std::string, const fb::TensorAttributes*> key{
        {"k", tensorWithUid(*graph, attributes->k_tensor_uid())},
        {"v", tensorWithUid(*graph, attributes->v_tensor_uid())},
        {"dk", tensorWithUid(*graph, attributes->dk_tensor_uid())},
        {"dv", tensorWithUid(*graph, attributes->dv_tensor_uid())}};
    const auto offsetOf = [](const fb::TensorAttributes* tensor) {
        return tensor == nullptr ? std::nullopt : tensor->ragged_offset_tensor_uid();
    };
    const auto queryOffsets = offsetOf(query.at("q"));
    const auto keyOffsets = offsetOf(key.at("k"));
    ASSERT_TRUE(queryOffsets.has_value());
    ASSERT_TRUE(keyOffsets.has_value());
    EXPECT_NE(*queryOffsets, *keyOffsets);
    for(const auto& [name, tensor] : query)
    {
        EXPECT_EQ(offsetOf(tensor), queryOffsets) << name;
    }
    for(const auto& [name, tensor] : key)
    {
        EXPECT_EQ(offsetOf(tensor), keyOffsets) << name;
    }
    for(const auto uid : {*queryOffsets, *keyOffsets})
    {
        const auto* offsets = tensorWithUid(*graph, uid);
        ASSERT_NE(offsets, nullptr);
        EXPECT_EQ(dimsOf(offsets), (Ints{BATCH + 1, 1, 1, 1}));
        EXPECT_EQ(offsets->data_type(), fb::DataType::INT32);
    }

    // Elements per token: each tensor's token stride, the statistics packed token-major.
    for(const auto* name : {"q", "o", "dO", "dq"})
    {
        EXPECT_EQ(query.at(name)->ragged_offset_multiplier(), HEADS * HEAD_DIM) << name;
    }
    EXPECT_EQ(query.at("stats")->ragged_offset_multiplier(), HEADS);
    for(const auto* name : {"k", "v", "dk", "dv"})
    {
        EXPECT_EQ(key.at(name)->ragged_offset_multiplier(), HEADS_KV * HEAD_DIM) << name;
    }
    EXPECT_EQ(graph->tensors()->size(), 13U);
}

TEST(TestSdpaForms, AFormTheDirectionDoesNotBuildIsRefusedByName)
{
    for(const auto* form : {"sink", "paged"})
    {
        const auto built = buildForm(test::shippedOperation("sdpa_bwd"), form);
        EXPECT_FALSE(built.ok()) << form;
        EXPECT_NE(built.error.find(form), std::string::npos) << built.error;
    }
    const auto built = buildForm(test::shippedOperation("sdpa_fwd"), "paged");
    EXPECT_FALSE(built.ok());
    EXPECT_NE(built.error.find("paged"), std::string::npos) << built.error;
}

namespace
{

/// @p metadata with graph-builder argument @p name removed, which it must have.
OperationMetadata withoutArgument(OperationMetadata metadata, const std::string& name)
{
    auto& arguments = metadata.graphBuilder.arguments;
    const auto found
        = std::find_if(arguments.begin(), arguments.end(), [&](const BuilderArgument& argument) {
              return argument.name == name;
          });
    EXPECT_NE(found, arguments.end()) << name;
    if(found != arguments.end())
    {
        arguments.erase(found);
    }
    return metadata;
}

/// @p metadata with graph-builder argument @p name fixed at @p value, which it must have.
OperationMetadata withConstantArgument(OperationMetadata metadata,
                                       const std::string& name,
                                       const nlohmann::json& value)
{
    auto& arguments = metadata.graphBuilder.arguments;
    const auto found
        = std::find_if(arguments.begin(), arguments.end(), [&](const BuilderArgument& argument) {
              return argument.name == name;
          });
    EXPECT_NE(found, arguments.end()) << name;
    if(found != arguments.end())
    {
        *found = BuilderArgument{};
        found->name = name;
        found->kind = BuilderArgument::Kind::CONSTANT;
        found->constant = value;
    }
    return metadata;
}

struct MissingOperandCase
{
    std::string operation;
    std::string form;
    std::string argument;
};

} // namespace

TEST(TestSdpaForms, AFormWhoseOperandIsNotDeclaredIsRefusedByFormAndArgument)
{
    // The dims, the strides and the element type are each required: an operand's type is part of
    // what it is, so none is defaulted from the graph's. (Dims a `strides_of` argument reads are
    // not removed: the resolver refuses that first.)
    const std::vector<MissingOperandCase> cases{
        {"sdpa_fwd", "scale_tensor", "scaleDims"},
        {"sdpa_fwd", "dropout", "seedDims"},
        {"sdpa_fwd", "padded", "seqLenStrides"},
        {"sdpa_fwd", "bias", "biasDataType"},
        {"sdpa_fwd", "sink", "sinkStrides"},
        {"sdpa_fwd", "ragged", "offsetsStrides"},
        {"sdpa_bwd", "scale_tensor", "scaleDataType"},
        {"sdpa_bwd", "dropout", "seedStrides"},
        {"sdpa_bwd", "padded", "seqLenDataType"},
        {"sdpa_bwd", "bias", "biasStrides"},
        {"sdpa_bwd", "ragged", "offsetsStrides"},
        {"sdpa_bwd", "ragged", "seqLenDataType"},
    };
    for(const auto& missing : cases)
    {
        const auto built = buildForm(
            withoutArgument(test::shippedOperation(missing.operation), missing.argument),
            missing.form);
        const auto where = missing.operation + " " + missing.form + " without " + missing.argument;
        EXPECT_FALSE(built.ok()) << where;
        EXPECT_NE(built.error.find("form '" + missing.form + "'"), std::string::npos)
            << where << ": " << built.error;
        EXPECT_NE(built.error.find(missing.argument), std::string::npos)
            << where << ": " << built.error;
    }
}

TEST(TestSdpaForms, AScaleTensorWithNoStatedScaleToDefaultToIsRefused)
{
    // The tensor's value is the stated scale, so a zero scale has nothing to give it.
    for(const auto& operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto built
            = buildForm(withConstantArgument(test::shippedOperation(operation), "attnScale", 0.0),
                        "scale_tensor");
        EXPECT_FALSE(built.ok()) << operation;
        EXPECT_NE(built.error.find("scale_tensor"), std::string::npos) << built.error;
        EXPECT_NE(built.error.find("attnScale"), std::string::npos) << built.error;
    }
}

TEST(TestSdpaForms, ARaggedFormOverTokensWithNoPositiveStrideIsRefused)
{
    // The stride is the offsets' elements per token; zero would pack every token onto one.
    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto built = buildForm(
            withConstantArgument(test::shippedOperation(operation), "qStrides", Ints{0, 0, 0, 1}),
            "ragged");
        EXPECT_FALSE(built.ok()) << operation;
        EXPECT_NE(built.error.find("ragged"), std::string::npos) << built.error;
        EXPECT_NE(built.error.find("qStrides"), std::string::npos) << built.error;
    }
}

namespace
{

const std::vector<std::string>& legacyDtypes()
{
    static const std::vector<std::string> dtypes{"fp16", "bf16", "fp32"};
    return dtypes;
}

const std::vector<std::string>& fp8Dtypes()
{
    static const std::vector<std::string> dtypes{
        "fp8_e4m3", "fp8_e5m2", "fp8_e4m3_fnuz", "fp8_e5m2_fnuz"};
    return dtypes;
}

bool isFp8Dtype(const std::string& dtype)
{
    return std::find(fp8Dtypes().begin(), fp8Dtypes().end(), dtype) != fp8Dtypes().end();
}

} // namespace

TEST(TestSdpaDtypes, TheForwardDeclaresEveryTypeItsGraphsCanCarryAndTheBackwardOnlyThree)
{
    // The forward takes the three element types and the four fp8 encodings, the latter with the
    // descales an fp8 plan requires. The backward takes the three only: its schema has no descale
    // operand to dequantize an fp8 input with (sdpa_backward_attributes.fbs), so an fp8 dtype
    // declared there would be a graph with nothing to read its inputs by. A dtype added to either
    // declaration needs its row in the "// types" note and an expectation here.
    const auto forward = test::shippedOperation("sdpa_fwd");
    const auto backward = test::shippedOperation("sdpa_bwd");
    const auto* forwardDtype = forward.find("dtype");
    const auto* backwardDtype = backward.find("dtype");
    ASSERT_NE(forwardDtype, nullptr);
    ASSERT_NE(backwardDtype, nullptr);

    std::set<std::string> everyType(legacyDtypes().begin(), legacyDtypes().end());
    everyType.insert(fp8Dtypes().begin(), fp8Dtypes().end());
    EXPECT_EQ(std::set<std::string>(forwardDtype->values.begin(), forwardDtype->values.end()),
              everyType);
    EXPECT_EQ(forwardDtype->values.size(), everyType.size()) << "a dtype is declared twice";
    EXPECT_EQ(backwardDtype->values, legacyDtypes());
    // The first value is what a test, and a shape file's missing column, takes as the default.
    EXPECT_EQ(forwardDtype->values.front(), "fp16");
}

TEST(TestSdpaDtypes, Fp8IsAMinorityOfTheForwardAndTheWiderTypesKeepTheirBalance)
{
    const auto metadata = test::shippedOperation("sdpa_fwd");
    const auto* dtype = metadata.find("dtype");
    ASSERT_NE(dtype, nullptr);
    // Unweighted, the four fp8 types would be four sevenths of the sweep.
    ASSERT_FALSE(dtype->weights.empty()) << "dtype is unweighted: fp8 would be 4/7 of the corpus";

    double total = 0.0;
    double fp8 = 0.0;
    for(const auto& value : dtype->values)
    {
        total += dtype->weightOf(value);
        fp8 += isFp8Dtype(value) ? dtype->weightOf(value) : 0.0;
    }
    EXPECT_LT(fp8 / total, 0.25);
    EXPECT_GT(fp8 / total, 0.1) << "too little fp8 for a model to learn from";
    // The types declared before fp8 stay in the proportion they had, one to one to one.
    EXPECT_EQ(dtype->weightOf("fp16"), dtype->weightOf("bf16"));
    EXPECT_EQ(dtype->weightOf("fp16"), dtype->weightOf("fp32"));
    // Each wider type outweighs every single fp8 encoding.
    for(const auto& value : fp8Dtypes())
    {
        EXPECT_LT(dtype->weightOf(value), dtype->weightOf("fp16")) << value;
    }
    // The backward weights nothing, as before.
    EXPECT_TRUE(test::shippedOperation("sdpa_bwd").find("dtype")->weights.empty());
}

TEST(TestSdpaDtypes, EveryForwardDtypeTypesItsTensorsAndBindsItsDescalesAsTheMatrixSays)
{
    // Every declared value, not a list here, so a dtype added to the declaration is held to the
    // rule without an edit: Q, K, V in the dtype; O (and the bias) in it for fp16, bf16 and fp32
    // and in bf16 for fp8; statistics FLOAT; for fp8 only, the three descales, each FLOAT, one
    // element, an ordinary device tensor; no mma_core_mode and none of the other fp8 scalings.
    const auto metadata = test::shippedOperation("sdpa_fwd");
    const auto* declared = metadata.find("dtype");
    ASSERT_NE(declared, nullptr);

    for(const auto& dtype : declared->values)
    {
        const bool fp8 = isFp8Dtype(dtype);
        const auto input = detail::dataTypeFor(dtype);
        ASSERT_TRUE(input.has_value()) << dtype;
        const auto output = fp8 ? fb::DataType::BFLOAT16 : *input;

        for(const auto& expected : formExpectations())
        {
            if(expected.forwardTensors == 0)
            {
                continue;
            }
            const auto where = dtype + " " + expected.form;
            const auto built = buildForm(metadata, expected.form, true, dtype);
            ASSERT_TRUE(built.ok()) << where << ": " << built.error;
            const auto* graph = asGraph(built.bytes);
            ASSERT_NE(graph, nullptr) << where;
            const auto* attributes = graph->nodes()->Get(0)->attributes_as_SdpaAttributes();
            ASSERT_NE(attributes, nullptr) << where;

            EXPECT_EQ(graph->io_data_type(), *input) << where;
            EXPECT_EQ(graph->compute_data_type(), fb::DataType::FLOAT) << where;
            EXPECT_EQ(graph->intermediate_data_type(), fb::DataType::FLOAT) << where;
            EXPECT_EQ(graph->nodes()->Get(0)->compute_data_type(), fb::DataType::FLOAT) << where;
            EXPECT_EQ(graph->tensors()->size(), expected.forwardTensors + (fp8 ? 3U : 0U)) << where;

            const auto typeOf = [&](std::optional<int64_t> uid) {
                const auto* tensor = tensorWithUid(*graph, uid);
                return tensor == nullptr ? fb::DataType::UNSET : tensor->data_type();
            };
            EXPECT_EQ(typeOf(attributes->q_tensor_uid()), *input) << where;
            EXPECT_EQ(typeOf(attributes->k_tensor_uid()), *input) << where;
            EXPECT_EQ(typeOf(attributes->v_tensor_uid()), *input) << where;
            EXPECT_EQ(typeOf(attributes->o_tensor_uid()), output) << where;
            EXPECT_EQ(typeOf(attributes->stats_tensor_uid()), fb::DataType::FLOAT) << where;
            if(expected.form == "bias")
            {
                EXPECT_EQ(typeOf(attributes->attn_mask_tensor_uid()), output) << where;
            }

            // Nothing the matrix leaves out: the softmax and output scaling, and the core mode.
            EXPECT_FALSE(attributes->descale_s_tensor_uid().has_value()) << where;
            EXPECT_FALSE(attributes->scale_s_tensor_uid().has_value()) << where;
            EXPECT_FALSE(attributes->scale_o_tensor_uid().has_value()) << where;
            EXPECT_FALSE(attributes->amax_s_tensor_uid().has_value()) << where;
            EXPECT_FALSE(attributes->amax_o_tensor_uid().has_value()) << where;
            EXPECT_EQ(attributes->mma_core_mode(), fb::DataType::UNSET) << where;

            const std::vector<std::optional<int64_t>> descales{attributes->descale_q_tensor_uid(),
                                                               attributes->descale_k_tensor_uid(),
                                                               attributes->descale_v_tensor_uid()};
            std::set<int64_t> descaleUids;
            for(const auto& uid : descales)
            {
                EXPECT_EQ(uid.has_value(), fp8) << where << ": descales belong to fp8 alone";
                if(!uid.has_value())
                {
                    continue;
                }
                descaleUids.insert(*uid);
                const auto* descale = tensorWithUid(*graph, uid);
                ASSERT_NE(descale, nullptr) << where;
                EXPECT_EQ(descale->data_type(), fb::DataType::FLOAT) << where;
                EXPECT_EQ(dimsOf(descale), (Ints{1, 1, 1, 1})) << where;
                // ASM reads a descale through a device pointer, so it is neither virtual nor
                // passed by value, whatever the references would take.
                EXPECT_FALSE(descale->virtual_()) << where;
                EXPECT_FALSE(descale->is_runtime_pass_by_value()) << where;
                EXPECT_EQ(descale->value_type(), fb::TensorValue::NONE) << where;
            }
            EXPECT_EQ(descaleUids.size(), fp8 ? 3U : 0U) << where << ": one tensor per descale";
        }
    }
}

TEST(TestSdpaDtypes, EveryBackwardDtypeTypesAllNineRolesAlike)
{
    // One element type across Q, K, V, O, dO and the three gradients, the statistics FLOAT, and
    // no descale or core mode: the backward schema has no field for either.
    const auto metadata = test::shippedOperation("sdpa_bwd");
    const auto* declared = metadata.find("dtype");
    ASSERT_NE(declared, nullptr);

    for(const auto& dtype : declared->values)
    {
        const auto input = detail::dataTypeFor(dtype);
        ASSERT_TRUE(input.has_value()) << dtype;
        for(const auto& form : {"plain", "bias"})
        {
            const auto where = dtype + " " + form;
            const auto built = buildForm(metadata, form, true, dtype);
            ASSERT_TRUE(built.ok()) << where << ": " << built.error;
            const auto* graph = asGraph(built.bytes);
            ASSERT_NE(graph, nullptr) << where;
            const auto* attributes = graph->nodes()->Get(0)->attributes_as_SdpaBackwardAttributes();
            ASSERT_NE(attributes, nullptr) << where;

            EXPECT_EQ(graph->io_data_type(), *input) << where;
            EXPECT_EQ(graph->compute_data_type(), fb::DataType::FLOAT) << where;
            for(const auto uid : {attributes->q_tensor_uid(),
                                  attributes->k_tensor_uid(),
                                  attributes->v_tensor_uid(),
                                  attributes->o_tensor_uid(),
                                  attributes->do_tensor_uid(),
                                  attributes->dq_tensor_uid(),
                                  attributes->dk_tensor_uid(),
                                  attributes->dv_tensor_uid()})
            {
                const auto* tensor = tensorWithUid(*graph, uid);
                ASSERT_NE(tensor, nullptr) << where;
                EXPECT_EQ(tensor->data_type(), *input) << where << " " << tensor->name()->str();
            }
            const auto* stats = tensorWithUid(*graph, attributes->stats_tensor_uid());
            ASSERT_NE(stats, nullptr) << where;
            EXPECT_EQ(stats->data_type(), fb::DataType::FLOAT) << where;
            EXPECT_EQ(std::string(form) == "bias" ? 10U : 9U, graph->tensors()->size()) << where;
        }
    }
}

TEST(TestSdpaDtypes, AnFp8ForwardWhoseDescaleIsNotDeclaredIsRefusedByArgumentAndOthersNeedNone)
{
    // The descales come with the dtype, so declaring fp8 without their dims, strides or type is
    // refused by name; an fp16 forward never reads them, which the declaration it already had
    // (no descale arguments) relies on.
    const auto metadata = test::shippedOperation("sdpa_fwd");
    const auto plain = buildForm(metadata, "plain", true, "fp16");
    ASSERT_TRUE(plain.ok()) << plain.error;
    for(const auto* argument : {"descaleDims", "descaleStrides", "descaleDataType"})
    {
        const auto without = withoutArgument(metadata, argument);
        const auto refused = buildForm(without, "plain", true, "fp8_e4m3_fnuz");
        EXPECT_FALSE(refused.ok()) << argument;
        EXPECT_NE(refused.error.find(argument), std::string::npos) << refused.error;
        EXPECT_NE(refused.error.find("fp8"), std::string::npos) << refused.error;

        const auto unaffected = buildForm(without, "plain", true, "fp16");
        ASSERT_TRUE(unaffected.ok()) << argument << ": " << unaffected.error;
        EXPECT_EQ(unaffected.bytes, plain.bytes) << argument;
    }

    // Byte for byte, for every form: the fp16, bf16 and fp32 graphs are what the declaration built
    // before it had the descale and output-type arguments, which an adapter that read them at any
    // dtype but fp8, or a map that touched a wider type, would change.
    auto before = metadata;
    for(const auto* argument : {"descaleDims", "descaleStrides", "descaleDataType", "oDataType"})
    {
        before = withoutArgument(before, argument);
    }
    for(const auto& dtype : legacyDtypes())
    {
        for(const auto& expected : formExpectations())
        {
            if(expected.forwardTensors == 0)
            {
                continue;
            }
            const auto now = buildForm(metadata, expected.form, true, dtype);
            const auto then = buildForm(before, expected.form, true, dtype);
            ASSERT_TRUE(now.ok()) << dtype << " " << expected.form << ": " << now.error;
            ASSERT_TRUE(then.ok()) << dtype << " " << expected.form << ": " << then.error;
            EXPECT_EQ(now.bytes, then.bytes) << dtype << " " << expected.form;
        }
    }
}

TEST(TestSdpaDtypes, TheOutputAndBiasSpellingsFollowTheDtypeAndAMisspeltOneIsRefused)
{
    const auto metadata = test::shippedOperation("sdpa_fwd");
    const auto spelled = [&metadata](const std::string& dtype, const char* argument) {
        const auto resolved
            = resolveArguments(metadata.graphBuilder, formPoint(metadata, "bias", true, dtype));
        EXPECT_TRUE(resolved.ok()) << resolved.error;
        const auto* found = resolved.find(argument);
        const auto* name = found == nullptr ? nullptr : std::get_if<std::string>(&found->value);
        return name == nullptr ? std::string("<unresolved>") : *name;
    };
    for(const auto& dtype : fp8Dtypes())
    {
        EXPECT_EQ(spelled(dtype, "dataType"), dtype);
        EXPECT_EQ(spelled(dtype, "oDataType"), "bf16") << dtype;
        EXPECT_EQ(spelled(dtype, "biasDataType"), "bf16") << dtype;
        EXPECT_EQ(spelled(dtype, "statsDataType"), "float") << dtype;
    }
    for(const auto& dtype : legacyDtypes())
    {
        EXPECT_EQ(spelled(dtype, "oDataType"), dtype);
        EXPECT_EQ(spelled(dtype, "biasDataType"), dtype);
    }

    const auto misspelt = buildForm(
        withConstantArgument(metadata, "oDataType", "bfloat17"), "plain", true, "fp8_e4m3_fnuz");
    EXPECT_FALSE(misspelt.ok());
    EXPECT_NE(misspelt.error.find("oDataType"), std::string::npos) << misspelt.error;
}

} // namespace hipdnn_corpus_gen

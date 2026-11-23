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

ProblemPoint formPoint(const std::string& form, bool stats = true)
{
    return ProblemPoint{{"batch", BATCH},
                        {"heads", HEADS},
                        {"heads_kv", HEADS_KV},
                        {"seqlen_q", SEQLEN_Q},
                        {"seqlen_k", SEQLEN_K},
                        {"head_dim", HEAD_DIM},
                        {"is_causal", true},
                        {"alignment", std::string("top_left")},
                        {"generate_stats", stats},
                        {"dtype", std::string("fp16")},
                        {"form", form}};
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
/// does not build the form has 0 tensors.
struct FormExpectation
{
    std::string form;
    std::set<std::string> addedRoles;
    std::set<std::string> addedScalars;
    std::set<std::string> removedScalars;
    size_t forwardTensors;
    size_t backwardTensors;
};

std::vector<FormExpectation> formExpectations()
{
    return {
        {"plain", {}, {}, {}, 5, 9},
        {"scale_unset", {}, {}, {"attn_scale_value"}, 5, 9},
        {"scale_tensor", {"scale"}, {}, {"attn_scale_value"}, 6, 10},
        {"dropout", {"seed", "offset"}, {"dropout_probability"}, {}, 7, 11},
        {"padded", {"seq_len_q", "seq_len_kv"}, {"padding_mask"}, {}, 7, 11},
        // The offsets go on the tensors, not the node.
        {"ragged", {}, {}, {}, 7, 0},
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

BuildResult buildForm(const OperationMetadata& metadata, const std::string& form, bool stats = true)
{
    return buildGraphFor(metadata, formPoint(form, stats));
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
    // The backward schema has no sink token, and no ragged form is built for it.
    expectEveryFormBindsExactlyItsOptionalFields(
        "sdpa_bwd", {"q", "k", "v", "o", "do", "stats", "dq", "dk", "dv"});
}

TEST(TestSdpaForms, ThePlainFormIsTheGraphTheDeclarationAlwaysBuilt)
{
    // No optional tensor, the stated scale: byte for byte what the builder writes given only
    // the operands, as it did before forms existed.
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
    const auto q = tensor(1,
                          "q",
                          {BATCH, HEADS, SEQLEN_Q, HEAD_DIM},
                          {SEQLEN_Q * HEADS * HEAD_DIM, HEAD_DIM, HEADS * HEAD_DIM, 1},
                          fb::DataType::HALF);
    const auto k = tensor(2,
                          "k",
                          {BATCH, HEADS_KV, SEQLEN_K, HEAD_DIM},
                          {SEQLEN_K * HEADS_KV * HEAD_DIM, HEAD_DIM, HEADS_KV * HEAD_DIM, 1},
                          fb::DataType::HALF);
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
    const builders::GraphTypes types{fb::DataType::HALF, fb::DataType::FLOAT, fb::DataType::FLOAT};

    const auto built = buildForm(metadata, "plain");
    ASSERT_TRUE(built.ok()) << built.error;
    EXPECT_EQ(built.bytes, builders::sdpaForward(q, k, v, o, options, types, stats));
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

TEST(TestSdpaForms, AFormTheDirectionDoesNotBuildIsRefusedByName)
{
    for(const auto* form : {"ragged", "sink", "paged"})
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
    const auto built = buildForm(
        withConstantArgument(test::shippedOperation("sdpa_fwd"), "qStrides", Ints{0, 0, 0, 1}),
        "ragged");
    EXPECT_FALSE(built.ok());
    EXPECT_NE(built.error.find("ragged"), std::string::npos) << built.error;
    EXPECT_NE(built.error.find("qStrides"), std::string::npos) << built.error;
}

} // namespace hipdnn_corpus_gen

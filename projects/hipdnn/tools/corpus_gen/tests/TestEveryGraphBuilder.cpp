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

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <algorithm>
#include <fstream>
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

OperationMetadata shippedAttention(const std::string& operation)
{
    const std::string path = test::operationsDir() + "/" + operation + ".opmeta.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());
    return parsed.metadata.value_or(OperationMetadata{});
}

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

/// Which optional fields of an SDPA node are set.
struct BoundFields
{
    bool attnScale = false;
    bool scaleTensor = false;
    bool dropout = false;
    bool seed = false;
    bool offset = false;
    bool padding = false;
    bool seqLenQ = false;
    bool seqLenKv = false;
    bool bias = false;
    bool sink = false;

    bool operator==(const BoundFields& other) const
    {
        return attnScale == other.attnScale && scaleTensor == other.scaleTensor
               && dropout == other.dropout && seed == other.seed && offset == other.offset
               && padding == other.padding && seqLenQ == other.seqLenQ && seqLenKv == other.seqLenKv
               && bias == other.bias && sink == other.sink;
    }
};

template <typename Attributes>
BoundFields boundFields(const Attributes& a)
{
    BoundFields bound;
    bound.attnScale = a.attn_scale_value().has_value();
    bound.scaleTensor = a.scale_tensor_uid().has_value();
    bound.dropout = a.dropout_probability().has_value();
    bound.seed = a.seed_tensor_uid().has_value();
    bound.offset = a.offset_tensor_uid().has_value();
    bound.padding = a.padding_mask();
    bound.seqLenQ = a.seq_len_q_tensor_uid().has_value();
    bound.seqLenKv = a.seq_len_kv_tensor_uid().has_value();
    bound.bias = a.attn_mask_tensor_uid().has_value();
    return bound;
}

BoundFields boundFields(const fb::SdpaAttributes& a)
{
    auto bound = boundFields<fb::SdpaAttributes>(a);
    bound.sink = a.sink_token_tensor_uid().has_value();
    return bound;
}

/// The fields each form binds, and how many tensors its graph holds (statistics included for
/// the forward; the backward's nine roles always).
struct FormExpectation
{
    std::string form;
    BoundFields bound;
    size_t forwardTensors;
    size_t backwardTensors;
};

std::vector<FormExpectation> formExpectations()
{
    BoundFields plain;
    plain.attnScale = true;
    const auto with = [&plain](auto edit) {
        auto bound = plain;
        edit(bound);
        return bound;
    };
    return {
        {"plain", plain, 5, 9},
        {"scale_unset", with([](auto& b) { b.attnScale = false; }), 5, 9},
        {"scale_tensor",
         with([](auto& b) {
             b.attnScale = false;
             b.scaleTensor = true;
         }),
         6,
         10},
        {"dropout",
         with([](auto& b) {
             b.dropout = true;
             b.seed = true;
             b.offset = true;
         }),
         7,
         11},
        {"padded",
         with([](auto& b) {
             b.padding = true;
             b.seqLenQ = true;
             b.seqLenKv = true;
         }),
         7,
         11},
        {"ragged", plain, 7, 0},
        {"bias", with([](auto& b) { b.bias = true; }), 6, 10},
        {"sink", with([](auto& b) { b.sink = true; }), 6, 0},
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

TEST(TestSdpaForms, EveryForwardFormBindsExactlyItsOptionalFields)
{
    const auto metadata = shippedAttention("sdpa_fwd");
    for(const auto& expected : formExpectations())
    {
        const auto built = buildForm(metadata, expected.form);
        ASSERT_TRUE(built.ok()) << expected.form << ": " << built.error;
        const auto* graph = asGraph(built.bytes);
        ASSERT_NE(graph, nullptr) << expected.form;
        const auto* attributes = graph->nodes()->Get(0)->attributes_as_SdpaAttributes();
        ASSERT_NE(attributes, nullptr) << expected.form;

        EXPECT_TRUE(boundFields(*attributes) == expected.bound) << expected.form;
        EXPECT_EQ(graph->tensors()->size(), expected.forwardTensors) << expected.form;
    }
}

TEST(TestSdpaForms, EveryBackwardFormBindsExactlyItsOptionalFields)
{
    const auto metadata = shippedAttention("sdpa_bwd");
    const auto* form = metadata.find("form");
    ASSERT_NE(form, nullptr);
    size_t built = 0;
    for(const auto& expected : formExpectations())
    {
        const bool declared = std::find(form->values.begin(), form->values.end(), expected.form)
                              != form->values.end();
        // The backward schema has no ragged offsets and no sink token.
        EXPECT_EQ(declared, expected.backwardTensors != 0) << expected.form;
        if(!declared)
        {
            continue;
        }
        const auto result = buildForm(metadata, expected.form);
        ASSERT_TRUE(result.ok()) << expected.form << ": " << result.error;
        const auto* graph = asGraph(result.bytes);
        ASSERT_NE(graph, nullptr) << expected.form;
        const auto* attributes = graph->nodes()->Get(0)->attributes_as_SdpaBackwardAttributes();
        ASSERT_NE(attributes, nullptr) << expected.form;

        EXPECT_TRUE(boundFields(*attributes) == expected.bound) << expected.form;
        EXPECT_EQ(graph->tensors()->size(), expected.backwardTensors) << expected.form;
        ++built;
    }
    EXPECT_EQ(built, 6U);
}

TEST(TestSdpaForms, ThePlainFormIsTheGraphTheDeclarationAlwaysBuilt)
{
    // No optional tensor, the stated scale: byte for byte what the builder writes given only
    // the operands, as it did before forms existed.
    const auto metadata = shippedAttention("sdpa_fwd");
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
    const auto built = buildForm(shippedAttention("sdpa_fwd"), "scale_tensor");
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
    const auto built = buildForm(shippedAttention("sdpa_fwd"), "dropout");
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
        const auto built = buildForm(shippedAttention(operation), "padded");
        ASSERT_TRUE(built.ok()) << operation << ": " << built.error;
        const auto* graph = asGraph(built.bytes);
        std::optional<int64_t> seqLenQ;
        std::optional<int64_t> seqLenKv;
        if(const auto* forward = graph->nodes()->Get(0)->attributes_as_SdpaAttributes())
        {
            seqLenQ = forward->seq_len_q_tensor_uid();
            seqLenKv = forward->seq_len_kv_tensor_uid();
        }
        else
        {
            const auto* backward = graph->nodes()->Get(0)->attributes_as_SdpaBackwardAttributes();
            seqLenQ = backward->seq_len_q_tensor_uid();
            seqLenKv = backward->seq_len_kv_tensor_uid();
        }
        for(const auto uid : {seqLenQ, seqLenKv})
        {
            const auto* tensor = tensorWithUid(*graph, uid);
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
        const auto built = buildForm(shippedAttention(operation), "bias");
        ASSERT_TRUE(built.ok()) << operation << ": " << built.error;
        const auto* graph = asGraph(built.bytes);
        const auto* node = graph->nodes()->Get(0);
        const auto uid = node->attributes_as_SdpaAttributes() != nullptr
                             ? node->attributes_as_SdpaAttributes()->attn_mask_tensor_uid()
                             : node->attributes_as_SdpaBackwardAttributes()->attn_mask_tensor_uid();
        const auto* bias = tensorWithUid(*graph, uid);
        ASSERT_NE(bias, nullptr) << operation;
        EXPECT_EQ(dimsOf(bias), (Ints{1, 1, SEQLEN_Q, SEQLEN_K})) << operation;
        EXPECT_EQ(bias->data_type(), fb::DataType::HALF) << operation;
    }
}

TEST(TestSdpaForms, TheSinkIsOneFloatLogitPerQueryHead)
{
    const auto built = buildForm(shippedAttention("sdpa_fwd"), "sink");
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
        const auto built = buildForm(shippedAttention("sdpa_fwd"), "ragged", stats);
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
        const auto built = buildForm(shippedAttention("sdpa_bwd"), form);
        EXPECT_FALSE(built.ok()) << form;
        EXPECT_NE(built.error.find(form), std::string::npos) << built.error;
    }
    const auto built = buildForm(shippedAttention("sdpa_fwd"), "paged");
    EXPECT_FALSE(built.ok());
    EXPECT_NE(built.error.find("paged"), std::string::npos) << built.error;
}

} // namespace hipdnn_corpus_gen

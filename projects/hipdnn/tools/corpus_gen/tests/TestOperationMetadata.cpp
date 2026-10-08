// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestOperationMetadata.cpp
 * @brief Covers the declared problem space and the argument mapping (RFC 0019.13 §4).
 *
 * A mapping written against a different parameterization still builds a graph from stale
 * defaults, mislabelled; §4.4 guards that. Uses §4.2's LayerNorm example verbatim.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/ArgumentResolver.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace hipdnn_corpus_gen
{
namespace
{

/// RFC 0019.13 §4.2's worked example, verbatim.
nlohmann::json layernormMetadata()
{
    return nlohmann::json::parse(R"({
      "schema_version": "1.0",
      "operation": "layernorm_fwd",
      "display_name": "LayerNorm Forward",
      "parameters": {
        "batch":         { "type": "int64" },
        "seq_len":       { "type": "int64" },
        "hidden_dim":    { "type": "int64" },
        "dtype":         { "type": "enum", "values": ["fp32", "fp16", "bf16"] },
        "forward_phase": { "type": "enum", "values": ["INFERENCE", "TRAINING"] }
      },
      "stratification_axis": "working_set",
      "regimes": {
        "hidden_dim": { "parameter": "hidden_dim", "buckets": [768, 1024, 4096] },
        "batch":      { "parameter": "batch", "buckets": [1, 4, 16] }
      },
      "graph_builder": {
        "function": "createValidLayernormFpropGraph",
        "source": "hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp",
        "arguments": [
          { "name": "dims", "kind": "expr",
            "value": ["$q.batch", "$q.seq_len", "$q.hidden_dim"] },
          { "name": "strides", "kind": "strides_of", "of": "dims" },
          { "name": "inputDataType", "kind": "dtype_of", "source": "$q.dtype" },
          { "name": "computeDataType", "kind": "dtype_of", "source": "$q.dtype" }
        ]
      }
    })");
}

} // namespace

TEST(TestOperationMetadata, LoadsTheWorkedExampleFromTheSpecification)
{
    const auto load = parseOperationMetadata(layernormMetadata());

    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());
    const auto& metadata = *load.metadata;

    EXPECT_EQ(metadata.operation, "layernorm_fwd");
    EXPECT_EQ(metadata.stratificationAxis, "working_set");
    EXPECT_EQ(metadata.parameters.size(), 5U);
    EXPECT_EQ(metadata.graphBuilder.function, "createValidLayernormFpropGraph");
    EXPECT_EQ(metadata.graphBuilder.arguments.size(), 4U);
}

TEST(TestOperationMetadata, EnumeratesTheCategoricalParametersAndNotTheNumericOnes)
{
    // dtype is part of the problem space, not a kernel property.
    const auto load = parseOperationMetadata(layernormMetadata());
    ASSERT_TRUE(load.ok());

    const auto* dtype = load.metadata->find("dtype");
    ASSERT_NE(dtype, nullptr);
    EXPECT_EQ(dtype->enumerable().size(), 3U);

    const auto* batch = load.metadata->find("batch");
    ASSERT_NE(batch, nullptr);
    EXPECT_TRUE(batch->enumerable().empty()) << "a numeric dimension is searched, not enumerated";
    EXPECT_FALSE(batch->range.has_value())
        << "§4.3.2: numeric dimensions are not bounded by authored ranges";
}

TEST(TestOperationMetadata, RejectsABuilderWrittenAgainstADifferentParameterization)
{
    // §4.4 check 2: otherwise this loads and builds from a default the mapping never overrides.
    auto broken = layernormMetadata();
    broken["graph_builder"]["arguments"][0]["value"] = {"$q.batch", "$q.sequence_length"};

    const auto load = parseOperationMetadata(broken);

    EXPECT_FALSE(load.ok());
    ASSERT_FALSE(load.errors.empty());
    EXPECT_NE(load.errors.front().find("sequence_length"), std::string::npos);
}

TEST(TestOperationMetadata, RejectsAForwardStridesReference)
{
    // Arguments resolve in declaration order; caught at load, not as a rank-zero tensor later.
    auto broken = layernormMetadata();
    broken["graph_builder"]["arguments"][1]["of"] = "not_yet_declared";

    const auto load = parseOperationMetadata(broken);

    EXPECT_FALSE(load.ok());
    EXPECT_NE(load.errors.front().find("not_yet_declared"), std::string::npos);
}

TEST(TestOperationMetadata, RejectsAnUnpermittedStratificationAxis)
{
    // §4.3.4 permits three axes; an unknown one would stratify nothing.
    auto broken = layernormMetadata();
    broken["stratification_axis"] = "flops";

    EXPECT_FALSE(parseOperationMetadata(broken).ok());
    EXPECT_TRUE(isPermittedStratificationAxis("arithmetic_intensity"));
    EXPECT_TRUE(isPermittedStratificationAxis("reduction_ratio"));
}

TEST(TestOperationMetadata, RejectsARegimeOverAnUndeclaredParameter)
{
    auto broken = layernormMetadata();
    broken["regimes"]["typo"] = {{"parameter", "hidden_size"}, {"buckets", {1, 2}}};

    const auto load = parseOperationMetadata(broken);
    EXPECT_FALSE(load.ok());
}

TEST(TestOperationMetadata, ResolvesTheWorkedExamplesArguments)
{
    const auto load = parseOperationMetadata(layernormMetadata());
    ASSERT_TRUE(load.ok());

    const ProblemPoint point{{"batch", int64_t{4}},
                             {"seq_len", int64_t{512}},
                             {"hidden_dim", int64_t{1024}},
                             {"dtype", std::string("fp16")},
                             {"forward_phase", std::string("TRAINING")}};

    const auto resolved = resolveArguments(load.metadata->graphBuilder, point);
    ASSERT_TRUE(resolved.ok()) << resolved.error;
    ASSERT_EQ(resolved.arguments.size(), 4U);

    EXPECT_EQ(std::get<std::vector<int64_t>>(resolved.arguments[0].value),
              (std::vector<int64_t>{4, 512, 1024}));

    // Row-major over the dims resolved immediately before, per §4.3.6.
    EXPECT_EQ(std::get<std::vector<int64_t>>(resolved.arguments[1].value),
              (std::vector<int64_t>{int64_t{512} * 1024, 1024, 1}));

    EXPECT_EQ(std::get<std::string>(resolved.arguments[2].value), "fp16");
    EXPECT_EQ(std::get<std::string>(resolved.arguments[3].value), "fp16");
}

TEST(TestOperationMetadata, RefusesToResolveAPointMissingAParameter)
{
    // Never defaulted: the graph would disagree with the row that labels it.
    const auto load = parseOperationMetadata(layernormMetadata());
    ASSERT_TRUE(load.ok());

    const ProblemPoint incomplete{{"batch", int64_t{4}}, {"seq_len", int64_t{512}}};
    const auto resolved = resolveArguments(load.metadata->graphBuilder, incomplete);

    EXPECT_FALSE(resolved.ok());
    EXPECT_NE(resolved.error.find("hidden_dim"), std::string::npos);
}

TEST(TestOperationMetadata, ComputesRowMajorStridesForAnyRank)
{
    EXPECT_EQ(detail::rowMajorStrides({2, 3, 4}), (std::vector<int64_t>{12, 4, 1}));
    EXPECT_EQ(detail::rowMajorStrides({5}), (std::vector<int64_t>{1}));
    EXPECT_EQ(detail::rowMajorStrides({2, 1, 4, 1}), (std::vector<int64_t>{4, 4, 1, 1}));
}

TEST(TestOperationMetadata, DefaultsToZerosWhenNoContentsAreDeclared)
{
    // Correct whenever the work is fixed by shape.
    const auto load = parseOperationMetadata(layernormMetadata());
    ASSERT_TRUE(load.ok());
    EXPECT_TRUE(load.metadata->variantPack.empty());
}

TEST(TestOperationMetadata, ReadsDeclaredTensorContents)
{
    // MoE routing lives in first_token_offset's contents, not in any dimension.
    auto metadata = layernormMetadata();
    metadata["parameters"]["num_experts"] = {{"type", "int64"}};
    metadata["parameters"]["skew"] = {{"type", "enum"}, {"values", {"uniform", "imbalanced"}}};
    metadata["variant_pack"]
        = nlohmann::json::array({{{"tensor", "first_token_offset"},
                                  {"fill", "routing_offsets"},
                                  {"arguments", {"$q.num_experts", "$q.skew"}}},
                                 {{"tensor", "token_index"},
                                  {"fill", "expert_assignment"},
                                  {"arguments", {"$q.num_experts"}}}});

    const auto load = parseOperationMetadata(metadata);

    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());
    ASSERT_EQ(load.metadata->variantPack.size(), 2U);
    EXPECT_EQ(load.metadata->variantPack[0].tensor, "first_token_offset");
    EXPECT_EQ(load.metadata->variantPack[0].kind, FillKind::ROUTING_OFFSETS);
    EXPECT_EQ(load.metadata->variantPack[0].arguments.size(), 2U);
}

TEST(TestOperationMetadata, RefusesAnUnknownFillRatherThanDefaultingToZeros)
{
    // Zeros would benchmark a routing the corpus never asked for.
    auto metadata = layernormMetadata();
    metadata["variant_pack"]
        = nlohmann::json::array({{{"tensor", "x"}, {"fill", "gaussian_with_outliers"}}});

    const auto load = parseOperationMetadata(metadata);

    EXPECT_FALSE(load.ok());
    EXPECT_NE(load.errors.front().find("gaussian_with_outliers"), std::string::npos);
}

TEST(TestOperationMetadata, RefusesAFillOverAnUndeclaredParameter)
{
    auto metadata = layernormMetadata();
    metadata["variant_pack"] = nlohmann::json::array(
        {{{"tensor", "x"}, {"fill", "routing_offsets"}, {"arguments", {"$q.experts"}}}});

    EXPECT_FALSE(parseOperationMetadata(metadata).ok());
}

TEST(TestOperationMetadata, ExpressesAConvolutionsOutputExtent)
{
    // Conv's output extent needs real arithmetic, evaluated by the shared §6.2 interpreter.
    const auto metadata = parseOperationMetadata(nlohmann::json::parse(R"({
      "schema_version": "1.0",
      "operation": "conv_fwd",
      "parameters": {
        "N": { "type": "int64" }, "C": { "type": "int64" }, "K": { "type": "int64" },
        "H": { "type": "int64" }, "W": { "type": "int64" },
        "R": { "type": "int64" }, "S": { "type": "int64" },
        "pad_h": { "type": "int64" }, "stride_h": { "type": "int64" },
        "dilation_h": { "type": "int64" },
        "dtype": { "type": "enum", "values": ["fp32", "fp16"] }
      },
      "stratification_axis": "arithmetic_intensity",
      "regimes": {},
      "graph_builder": {
        "function": "createValidConvFwdGraph",
        "source": "hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp",
        "arguments": [
          { "name": "yDims", "kind": "expr", "value": [
              "$q.N",
              "$q.K",
              { "+": [ { "/": [ { "-": [ { "-": [ { "+": ["$q.H", { "*": [2, "$q.pad_h"] }] },
                                                  { "*": ["$q.dilation_h",
                                                          { "-": ["$q.R", 1] }] } ] },
                                        1 ] },
                                "$q.stride_h" ] },
                       1 ] }
          ] }
        ]
      }
    })"));

    ASSERT_TRUE(metadata.ok()) << (metadata.errors.empty() ? "" : metadata.errors.front());

    // (224 + 2*3 - 1*(7-1) - 1)/2 + 1 = 112. ResNet50 conv1.
    const ProblemPoint resnetConv1{{"N", int64_t{64}},
                                   {"C", int64_t{3}},
                                   {"K", int64_t{64}},
                                   {"H", int64_t{224}},
                                   {"W", int64_t{224}},
                                   {"R", int64_t{7}},
                                   {"S", int64_t{7}},
                                   {"pad_h", int64_t{3}},
                                   {"stride_h", int64_t{2}},
                                   {"dilation_h", int64_t{1}},
                                   {"dtype", std::string("fp16")}};

    const auto resolved = resolveArguments(metadata.metadata->graphBuilder, resnetConv1);
    ASSERT_TRUE(resolved.ok()) << resolved.error;
    EXPECT_EQ(std::get<std::vector<int64_t>>(resolved.arguments[0].value),
              (std::vector<int64_t>{64, 64, 112}));
}

TEST(TestOperationMetadata, ANestedExpressionsVariablesAreStillChecked)
{
    // §4.4 check 2 must see variables nested inside an expression.
    auto broken = nlohmann::json::parse(R"({
      "schema_version": "1.0",
      "operation": "conv_fwd",
      "parameters": { "H": { "type": "int64" } },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [
        { "name": "dims", "kind": "expr",
          "value": [ { "+": [ { "*": [2, "$q.heigth"] }, 1 ] } ] } ] }
    })");

    const auto load = parseOperationMetadata(broken);
    EXPECT_FALSE(load.ok());
    EXPECT_NE(load.errors.front().find("heigth"), std::string::npos);
}

namespace
{

/// An operation with a kernel pool; `catalog` is spliced in as its `kernel_catalog` block.
nlohmann::json catalogMetadata(const nlohmann::json& catalog)
{
    auto declaration = nlohmann::json::parse(R"({
      "schema_version": "1.0",
      "operation": "sdpa_fwd",
      "parameters": {
        "batch":     { "type": "int64" },
        "is_causal": { "type": "bool" },
        "scale":     { "type": "float64" },
        "alignment": { "type": "enum", "values": ["top_left", "bottom_right"] },
        "dtype":     { "type": "enum", "values": ["bf16", "fp16"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] }
    })");
    declaration["kernel_catalog"] = catalog;
    return declaration;
}

} // namespace

TEST(TestOperationMetadata, AConstantAnswersForAParameterThePackVocabularyHasNoFieldFor)
{
    // rocKE's dense packs omit the causal anchor (always top-left); a constant supplies it.
    const auto load = parseOperationMetadata(catalogMetadata(nlohmann::json::parse(R"({
      "metadata": { "batch": "batch", "is_causal": "causal" },
      "constants": { "alignment": "top_left", "batch_size_hint": 0 }
    })")));

    // `batch_size_hint` is undeclared: a constant for an undeclared parameter is a typo.
    EXPECT_FALSE(load.ok());
    EXPECT_NE(load.errors.front().find("batch_size_hint"), std::string::npos)
        << load.errors.front();
}

TEST(TestOperationMetadata, AConstantIsTypedAgainstItsParameterAtLoad)
{
    // Typed at load: point construction has no error channel.
    const auto wrongType = [](const char* json) {
        const auto load = parseOperationMetadata(catalogMetadata(nlohmann::json::parse(json)));
        EXPECT_FALSE(load.ok()) << json;
        return load.ok() ? std::string{} : load.errors.front();
    };

    EXPECT_NE(wrongType(R"({"metadata": {"batch": "batch"}, "constants": {"is_causal": 1}})")
                  .find("not a boolean"),
              std::string::npos);
    EXPECT_NE(wrongType(R"({"metadata": {"batch": "batch"}, "constants": {"scale": "half"}})")
                  .find("not a number"),
              std::string::npos);
    EXPECT_NE(wrongType(R"({"metadata": {"batch": "batch"}, "constants": {"alignment": "middle"}})")
                  .find("not one of its declared values"),
              std::string::npos);

    const auto load = parseOperationMetadata(catalogMetadata(nlohmann::json::parse(R"({
      "metadata": { "batch": "batch" },
      "constants": { "is_causal": true, "scale": 0.125, "alignment": "bottom_right" }
    })")));
    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());

    const auto& constants = load.metadata->kernelCatalog.constants;
    EXPECT_TRUE(std::get<bool>(constants.at("is_causal")));
    EXPECT_DOUBLE_EQ(std::get<double>(constants.at("scale")), 0.125);
    EXPECT_EQ(std::get<std::string>(constants.at("alignment")), "bottom_right");
}

TEST(TestOperationMetadata, AConstantMayNotOverrideWhatAPackActuallySaid)
{
    // A constant must not replace a geometry field the pack actually supplies.
    const auto load = parseOperationMetadata(catalogMetadata(nlohmann::json::parse(R"({
      "metadata": { "batch": "batch", "is_causal": "causal" },
      "constants": { "is_causal": true }
    })")));

    EXPECT_FALSE(load.ok());
    EXPECT_NE(load.errors.front().find("which it also reads from the pack"), std::string::npos)
        << load.errors.front();
}

TEST(TestOperationMetadata, AConditionalDimsElementIsOmittedWhenItsConditionIsFalse)
{
    // One declaration, two ranks: D is present for 3-D problems and absent for 2-D ones.
    auto declaration = layernormMetadata();
    declaration["parameters"]["spatial"] = {{"type", "enum"}, {"values", {"2d", "3d"}}};
    declaration["graph_builder"]["arguments"][0]["value"]
        = nlohmann::json::array({"$q.batch",
                                 {{"when", {{"==", {"$q.spatial", "3d"}}}}, {"value", 7}},
                                 "$q.seq_len",
                                 "$q.hidden_dim"});
    const auto load = parseOperationMetadata(declaration);
    ASSERT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());

    ProblemPoint point{{"batch", int64_t{4}},
                       {"seq_len", int64_t{512}},
                       {"hidden_dim", int64_t{1024}},
                       {"dtype", std::string("fp16")},
                       {"forward_phase", std::string("TRAINING")},
                       {"spatial", std::string("3d")}};
    auto resolved = resolveArguments(load.metadata->graphBuilder, point);
    ASSERT_TRUE(resolved.ok()) << resolved.error;
    EXPECT_EQ(std::get<std::vector<int64_t>>(resolved.arguments[0].value),
              (std::vector<int64_t>{4, 7, 512, 1024}));

    point["spatial"] = std::string("2d");
    resolved = resolveArguments(load.metadata->graphBuilder, point);
    ASSERT_TRUE(resolved.ok()) << resolved.error;
    EXPECT_EQ(std::get<std::vector<int64_t>>(resolved.arguments[0].value),
              (std::vector<int64_t>{4, 512, 1024}));
}

} // namespace hipdnn_corpus_gen

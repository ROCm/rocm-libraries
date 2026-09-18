// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestOperationMetadata.cpp
 * @brief Covers the declared problem space and the argument mapping (RFC 0019.13 §4).
 *
 * The failures here are quiet ones. A builder mapping written against a different
 * parameterization than the operation declares still builds a graph — from stale defaults —
 * and that graph benchmarks, produces a time, and gets labelled with parameters that never
 * reached it. §4.4 exists for that case, and so does most of this file.
 *
 * The metadata under test is the LayerNorm example from §4.2, used verbatim so a change to
 * the RFC's own example shows up as a failure here rather than as drift.
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
        "dtype":         { "type": "enum", "values": ["float32", "float16", "bfloat16"] },
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
    // dtype is part of the problem space, not a kernel property. Hardcoding it -- as an earlier
    // version of this generator did -- makes every problem float32 and leaves the engine's
    // half-precision kernels unreachable, with nothing in the corpus saying so.
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
    // §4.4 check 2, and the reason it is worth having: this metadata is otherwise well formed,
    // so without the check it loads, builds graphs from a default the mapping never overrides,
    // and yields a corpus whose rows describe problems that were never run.
    auto broken = layernormMetadata();
    broken["graph_builder"]["arguments"][0]["value"] = {"$q.batch", "$q.sequence_length"};

    const auto load = parseOperationMetadata(broken);

    EXPECT_FALSE(load.ok());
    ASSERT_FALSE(load.errors.empty());
    EXPECT_NE(load.errors.front().find("sequence_length"), std::string::npos);
}

TEST(TestOperationMetadata, RejectsAForwardStridesReference)
{
    // Arguments resolve in declaration order, so strides_of naming a later argument cannot be
    // satisfied. Caught at load rather than at resolve, where it would read as an empty dims
    // list and produce a rank-zero tensor that fails somewhere unrelated.
    auto broken = layernormMetadata();
    broken["graph_builder"]["arguments"][1]["of"] = "not_yet_declared";

    const auto load = parseOperationMetadata(broken);

    EXPECT_FALSE(load.ok());
    EXPECT_NE(load.errors.front().find("not_yet_declared"), std::string::npos);
}

TEST(TestOperationMetadata, RejectsAnUnpermittedStratificationAxis)
{
    // §4.3.4 permits three. Arithmetic intensity is shape-invariant for a bandwidth-bound op,
    // so the axis is declared rather than assumed, and an unrecognised one would silently
    // stratify nothing.
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
                             {"dtype", std::string("float16")},
                             {"forward_phase", std::string("TRAINING")}};

    const auto resolved = resolveArguments(load.metadata->graphBuilder, point);
    ASSERT_TRUE(resolved.ok()) << resolved.error;
    ASSERT_EQ(resolved.arguments.size(), 4U);

    EXPECT_EQ(std::get<std::vector<int64_t>>(resolved.arguments[0].value),
              (std::vector<int64_t>{4, 512, 1024}));

    // Row-major over the dims resolved immediately before, per §4.3.6.
    EXPECT_EQ(std::get<std::vector<int64_t>>(resolved.arguments[1].value),
              (std::vector<int64_t>{int64_t{512} * 1024, 1024, 1}));

    EXPECT_EQ(std::get<std::string>(resolved.arguments[2].value), "float16");
    EXPECT_EQ(std::get<std::string>(resolved.arguments[3].value), "float16");
}

TEST(TestOperationMetadata, RefusesToResolveAPointMissingAParameter)
{
    // Not a default. A missing parameter silently substituted would produce a graph whose
    // shape disagrees with the row that labels it, which is the one failure this whole layer
    // is arranged to prevent.
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
    // Correct for every operation whose work is fixed by shape, which is most of them. The
    // declaration exists for the ones where it is not.
    const auto load = parseOperationMetadata(layernormMetadata());
    ASSERT_TRUE(load.ok());
    EXPECT_TRUE(load.metadata->variantPack.empty());
}

TEST(TestOperationMetadata, ReadsDeclaredTensorContents)
{
    // An MoE grouped matmul's routing lives in the contents of first_token_offset, not in any
    // dimension. Two problems with identical graphs and different routing do different work,
    // so the contents are part of the problem specification.
    auto metadata = layernormMetadata();
    metadata["parameters"]["num_experts"] = {{"type", "int64"}};
    metadata["parameters"]["skew"]
        = {{"type", "enum"}, {"values", {"uniform", "imbalanced"}}};
    metadata["variant_pack"] = nlohmann::json::array(
        {{{"tensor", "first_token_offset"},
          {"fill", "routing_offsets"},
          {"arguments", {"$q.num_experts", "$q.skew"}}},
         {{"tensor", "token_index"}, {"fill", "expert_assignment"},
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
    // Substituting zeros for a fill nobody can produce is precisely the failure the
    // declaration exists to prevent: the benchmark still runs, still produces a time, and the
    // time describes a routing the corpus never asked for.
    auto metadata = layernormMetadata();
    metadata["variant_pack"] = nlohmann::json::array(
        {{{"tensor", "x"}, {"fill", "gaussian_with_outliers"}}});

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
    // The case that forced expression evaluation to be real. A convolution's y extent is
    // (H + 2*pad - dilation*(R-1) - 1)/stride + 1, so a resolver that understood only variable
    // references and literals could not describe conv at all -- which is how a hand-written
    // C++ builder ends up existing for it.
    //
    // Evaluated by the shared §6.2 interpreter, the same one UMD criteria and UDD dispatch
    // formulas use, rather than by arithmetic local to the corpus generator.
    const auto metadata = parseOperationMetadata(nlohmann::json::parse(R"({
      "schema_version": "1.0",
      "operation": "conv_fwd",
      "parameters": {
        "N": { "type": "int64" }, "C": { "type": "int64" }, "K": { "type": "int64" },
        "H": { "type": "int64" }, "W": { "type": "int64" },
        "R": { "type": "int64" }, "S": { "type": "int64" },
        "pad_h": { "type": "int64" }, "stride_h": { "type": "int64" },
        "dilation_h": { "type": "int64" },
        "dtype": { "type": "enum", "values": ["float32", "float16"] }
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

    // 224 + 2*3 - 1*(7-1) - 1 = 223; 223/2 = 111; + 1 = 112. ResNet50 conv1.
    const ProblemPoint resnetConv1{{"N", int64_t{64}},   {"C", int64_t{3}},
                                   {"K", int64_t{64}},   {"H", int64_t{224}},
                                   {"W", int64_t{224}},  {"R", int64_t{7}},
                                   {"S", int64_t{7}},    {"pad_h", int64_t{3}},
                                   {"stride_h", int64_t{2}}, {"dilation_h", int64_t{1}},
                                   {"dtype", std::string("float16")}};

    const auto resolved = resolveArguments(metadata.metadata->graphBuilder, resnetConv1);
    ASSERT_TRUE(resolved.ok()) << resolved.error;
    EXPECT_EQ(std::get<std::vector<int64_t>>(resolved.arguments[0].value),
              (std::vector<int64_t>{64, 64, 112}));
}

TEST(TestOperationMetadata, ANestedExpressionsVariablesAreStillChecked)
{
    // §4.4 check 2 has to see inside an expression, not just scan its text. The variables come
    // from the evaluator, so a typo buried three levels deep is caught at load rather than
    // producing a graph built from whatever the interpreter did with an unbound symbol.
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

} // namespace hipdnn_corpus_gen

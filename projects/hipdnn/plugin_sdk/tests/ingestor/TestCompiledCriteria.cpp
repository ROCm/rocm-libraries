// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledCriteria.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPatternMatcher.hpp>

#include "GraphPatternTestSupport.hpp"

/// @file TestCompiledCriteria.cpp
/// @brief A UMD's declarative `criteria` over a binding a compiled `graph_match.nodes`
///        pattern published: compilation, evaluation, and the per-engine reference check.

namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
using hipdnn_plugin_sdk::ingestor::testing::BINARY_POINTWISE_PATTERN;
using hipdnn_plugin_sdk::ingestor::testing::compilePatternText;
using hipdnn_plugin_sdk::ingestor::testing::PatternGraphBuilder;
using hipdnn_plugin_sdk::ingestor::testing::pointwiseNode;

/// BINARY_POINTWISE_PATTERN with `in_1` optional.
const std::string OPTIONAL_SECOND_OPERAND = R"([{"kind": "op", "id": "pointwise",
    "op": "pointwise", "operands": {"in_0": "$input_a", "in_1": "$input_b?"},
    "results": {"out_0": "$output"}}])";

std::shared_ptr<const CompiledCriteria> criteria(const std::string& text,
                                                 MatchScope scope = MatchScope::GRAPH)
{
    return compileCriteria(nlohmann::json::parse(text), scope, "criteria");
}

/// The binding @p pattern publishes for @p graph; the pattern must accept it.
BoundTokens bindPattern(const std::string& pattern, const PatternGraphBuilder& graph)
{
    const auto compiled = compilePatternText(pattern);
    const auto built = graph.build();
    auto bound = matchGraphPattern(*compiled, built.context());
    if(!bound.has_value())
    {
        throw std::runtime_error("the pattern declined the test graph");
    }
    return *bound;
}

/// `t3 = t1 + t2` over 1x1x1x1 FLOAT tensors.
BoundTokens addBinding()
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).node(pointwiseNode(PointwiseMode::ADD, 1, 2, 3));
    return bindPattern(BINARY_POINTWISE_PATTERN, graph);
}

/// `t3 = abs(t1)`, binding OPTIONAL_SECOND_OPERAND with `input_b` absent.
BoundTokens unaryBinding()
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(3).node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 3));
    return bindPattern(OPTIONAL_SECOND_OPERAND, graph);
}

bool passes(const std::string& text, const BoundTokens& bound)
{
    const CriteriaDevice device{DeviceProperties{}};
    return criteria(text)->passes(CriteriaSymbols(bound, device));
}

KernelDefinition kernelWith(MetadataValues metadata)
{
    KernelDefinition kernel;
    kernel.metadata = std::move(metadata);
    return kernel;
}

MetadataSchema schemaWith(std::vector<MetadataField> fields)
{
    MetadataSchema schema;
    schema.name = "test schema";
    schema.fields = std::move(fields);
    return schema;
}

std::string compileError(const std::string& text, MatchScope scope)
{
    try
    {
        criteria(text, scope);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        return error.what();
    }
    return {};
}

} // namespace

TEST(TestCompiledCriteria, RefusesALiteralRoot)
{
    // A literal states no check: `true` would admit every graph.
    EXPECT_NE(compileError("true", MatchScope::GRAPH).find("not a literal"), std::string::npos);
    EXPECT_NE(compileError(R"("ADD")", MatchScope::GRAPH).find("not a literal"), std::string::npos);
    // `$$` escapes a literal string; it is not a reference.
    EXPECT_NE(compileError(R"("$$input_a")", MatchScope::GRAPH).find("not a literal"),
              std::string::npos);
}

TEST(TestCompiledCriteria, RefusesAnExpressionTheLanguageRejects)
{
    const auto message = compileError(R"({"hipdnn.custom": ["$input_a"]})", MatchScope::GRAPH);
    EXPECT_NE(message.find("does not compile"), std::string::npos) << message;
}

TEST(TestCompiledCriteria, RefusesAGraphScopedCriterionReadingKernelMetadata)
{
    // A graph-scoped verdict prunes the whole pack, so it cannot rest on one kernel's field.
    const auto message
        = compileError(R"({"==": ["$input_a.dtype", "$kernel.dtype"]})", MatchScope::GRAPH);
    EXPECT_NE(message.find("$kernel.dtype"), std::string::npos) << message;
    EXPECT_TRUE(
        compileError(R"({"==": ["$input_a.dtype", "$kernel.dtype"]})", MatchScope::KERNEL).empty());
}

TEST(TestCompiledCriteria, ReadsEveryKindOfFieldThePatternPublishes)
{
    const auto bound = addBinding();

    EXPECT_TRUE(passes(R"({"and": [
        {"==": ["$input_a.dtype", "FLOAT"]},
        {"==": ["$input_a.rank", 4]},
        {"==": ["$input_b.dims[3]", 1]},
        {"==": ["$output.strides[0]", 1]},
        "$input_a.packed",
        {"!": ["$output.virtual"]},
        {"==": ["$input_a", "$input_a.uid"]},
        {"==": ["$pointwise.operation", "ADD"]},
        {"==": ["$graph.node_count", 1]}]})",
                       bound));
    EXPECT_FALSE(passes(R"({"==": ["$pointwise.operation", "MUL"]})", bound));
}

TEST(TestCompiledCriteria, ComparesAPublishedStrideOrderAgainstALayoutAlias)
{
    // The binding holds stride_order as an integer list, which the alias expands to match.
    const auto bound = addBinding();

    EXPECT_TRUE(passes(R"({"==": ["$input_a.stride_order", "nchw"]})", bound));
    EXPECT_TRUE(passes(R"({"==": ["$input_a.stride_order", [3, 2, 1, 0]]})", bound));
    EXPECT_FALSE(passes(R"({"==": ["$input_a.stride_order", "nhwc"]})", bound));
}

TEST(TestCompiledCriteria, AnAxisTheTensorLacksDeclinesAndSoDoesItsNegation)
{
    const auto bound = addBinding();

    EXPECT_FALSE(passes(R"({"==": ["$input_a.dims[7]", 1]})", bound));
    EXPECT_FALSE(passes(R"({"!": [{"==": ["$input_a.dims[7]", 1]}]})", bound));
}

TEST(TestCompiledCriteria, AnAbsentOptionalOperandIsAnsweredOnlyByThePresenceOperators)
{
    const auto bound = unaryBinding();

    EXPECT_TRUE(passes(R"({"not_present": ["$input_b"]})", bound));
    EXPECT_FALSE(passes(R"({"present": ["$input_b"]})", bound));
    EXPECT_TRUE(passes(R"({"or": [{"not_present": ["$input_b"]},
                                  {"==": ["$input_b.dtype", "FLOAT"]}]})",
                       bound));
    // A field read on the absent operand is unknown, and so is its negation.
    EXPECT_FALSE(passes(R"({"==": ["$input_b.dtype", "FLOAT"]})", bound));
    EXPECT_FALSE(passes(R"({"!=": ["$input_b.dtype", "FLOAT"]})", bound));
}

TEST(TestCompiledCriteria, AResultThatIsNotABooleanDeclines)
{
    const auto bound = addBinding();

    // Truthy, but a criterion is Bool-rooted: the author compared nothing.
    EXPECT_FALSE(passes(R"("$input_a.rank")", bound));
    EXPECT_FALSE(passes(R"({"+": ["$input_a.rank", 1]})", bound));
    EXPECT_TRUE(passes(R"("$input_a.packed")", bound));
}

TEST(TestCompiledCriteria, AKernelScopedCriterionReadsTheCandidatesMetadata)
{
    const auto bound = addBinding();
    const CriteriaDevice device{DeviceProperties{}};
    const auto compiled = criteria(R"({"and": [{"==": ["$input_a.dtype", "$kernel.dtype"]},
                                               {"==": ["$kernel.tile[1]", 32]}]})",
                                   MatchScope::KERNEL);

    const auto floatKernel
        = kernelWith({{"dtype", std::string("FLOAT")}, {"tile", std::vector<int64_t>{64, 32}}});
    const auto halfKernel
        = kernelWith({{"dtype", std::string("HALF")}, {"tile", std::vector<int64_t>{64, 32}}});
    const auto shortTile
        = kernelWith({{"dtype", std::string("FLOAT")}, {"tile", std::vector<int64_t>{64}}});

    EXPECT_TRUE(compiled->passes(CriteriaSymbols(bound, device, &floatKernel)));
    EXPECT_FALSE(compiled->passes(CriteriaSymbols(bound, device, &halfKernel)));
    EXPECT_FALSE(compiled->passes(CriteriaSymbols(bound, device, &shortTile)));
}

TEST(TestCompiledCriteria, ReadsDevicePropertiesAndDeclinesOnOnesTheDeviceDidNotReport)
{
    const BoundTokens bound;
    const auto compiled
        = criteria(R"({"<=": ["$kernel.lds", "$device.lds_size"]})", MatchScope::KERNEL);
    const auto fits = kernelWith({{"lds", int64_t{32768}}});
    const auto overflows = kernelWith({{"lds", int64_t{131072}}});

    DeviceProperties reported;
    reported.sharedMemPerBlock = 65536;
    const CriteriaDevice device(reported);
    EXPECT_TRUE(compiled->passes(CriteriaSymbols(bound, device, &fits)));
    EXPECT_FALSE(compiled->passes(CriteriaSymbols(bound, device, &overflows)));

    // An unreported LDS size reads 0. It must not be compared as one, in either direction.
    const CriteriaDevice unreported{DeviceProperties{}};
    const auto above
        = criteria(R"({">": ["$kernel.lds", "$device.lds_size"]})", MatchScope::KERNEL);
    EXPECT_FALSE(compiled->passes(CriteriaSymbols(bound, unreported, &fits)));
    EXPECT_FALSE(above->passes(CriteriaSymbols(bound, unreported, &fits)));
}

TEST(TestCompiledCriteria, EveryReferenceToWhatThePatternPublishesResolves)
{
    const GraphMatch pattern{compilePatternText(OPTIONAL_SECOND_OPERAND)};
    const auto schema = schemaWith({{"dtype", MetadataType::STRING, std::nullopt},
                                    {"tile", MetadataType::INT_LIST, std::nullopt}});
    const auto compiled = criteria(R"({"and": [
        {"==": ["$input_a.dtype", "$kernel.dtype"]},
        {"==": ["$kernel.tile[0]", 64]},
        {"==": ["$input_a.dims[12]", "$output.strides[0]"]},
        {"==": ["$input_a.stride_order", "nchw"]},
        {"in": ["$input_a.rank", [4, 5]]},
        "$input_a.packed", {"!": ["$output.virtual"]}, {"!": ["$output.is_runtime_pass_by_value"]},
        {"present": ["$input_a.value_f32"]}, {"==": ["$output", "$output.uid"]},
        {"or": [{"not_present": ["$input_b"]}, {"==": ["$input_b.dtype", "FLOAT"]}]},
        {"==": ["$pointwise.operation", "ADD"]}, {"present": ["$pointwise.axis_tensor_uid"]},
        {"==": ["$pointwise.compute_data_type", "FLOAT"]},
        {"==": ["$graph.node_count", 1]}, {"!": ["$graph.is_override_shape_enabled"]},
        {"<=": ["$kernel.tile[1]", "$device.lds_size"]}]})",
                                   MatchScope::KERNEL);

    EXPECT_EQ(criteriaReferenceError(*compiled, pattern, schema), "");
}

struct UnresolvedReferenceCase
{
    const char* name;
    const char* criteria;
    const char* reported; ///< What the diagnostic must name.
};

class TestCompiledCriteriaUnresolved : public ::testing::TestWithParam<UnresolvedReferenceCase>
{
};

/// Each reference names something the pattern, the KMD, or the device does not publish,
/// so the pair is refused at load rather than declining on every graph.
TEST_P(TestCompiledCriteriaUnresolved, RefusesTheReferenceNamingIt)
{
    const GraphMatch pattern{compilePatternText(OPTIONAL_SECOND_OPERAND)};
    const auto schema = schemaWith({{"dtype", MetadataType::STRING, std::nullopt},
                                    {"tile", MetadataType::INT_LIST, std::nullopt}});
    const auto compiled = criteria(GetParam().criteria, MatchScope::KERNEL);

    const auto error = criteriaReferenceError(*compiled, pattern, schema);

    EXPECT_NE(error.find(GetParam().reported), std::string::npos) << error;
}

INSTANTIATE_TEST_SUITE_P(
    References,
    TestCompiledCriteriaUnresolved,
    ::testing::Values(
        UnresolvedReferenceCase{
            "UnboundVariable", R"({"==": ["$weights.dtype", "FLOAT"]})", "$weights.dtype"},
        UnresolvedReferenceCase{
            "UnpublishedTensorField", R"({"==": ["$input_a.layout", 1]})", "$input_a.layout"},
        // tensorElement() never writes a leading zero, so this token is never published.
        UnresolvedReferenceCase{
            "NonCanonicalIndex", R"({"==": ["$input_a.dims[01]", 1]})", "$input_a.dims[01]"},
        UnresolvedReferenceCase{
            "UnpublishedGraphField", R"({"==": ["$graph.edges", 1]})", "$graph.edges"},
        UnresolvedReferenceCase{
            "UnknownDeviceProperty", R"({"==": ["$device.arch", "gfx942"]})", "$device.arch"},
        UnresolvedReferenceCase{
            "UnpublishedAttribute", R"({"==": ["$pointwise.mode", "ADD"]})", "$pointwise.mode"},
        UnresolvedReferenceCase{
            "UndeclaredKernelField", R"({"==": ["$kernel.tile_n", 1]})", "$kernel.tile_n"},
        UnresolvedReferenceCase{"IndexIntoAScalarKernelField",
                                R"({"==": ["$kernel.dtype[0]", "F"]})",
                                "$kernel.dtype[0]"},
        // The pattern fixes the answer: a required operand is always present.
        UnresolvedReferenceCase{
            "PresenceOfARequiredOperand", R"({"present": ["$input_a"]})", "$input_a"},
        UnresolvedReferenceCase{"PresenceOfARequiredAttribute",
                                R"({"not_present": ["$pointwise.operation"]})",
                                "$pointwise.operation"},
        // A presence test nested under another operator is found as well.
        UnresolvedReferenceCase{"NestedPresenceOfARequiredOperand",
                                R"({"and": [{"==": ["$input_a.rank", 4]},
                                            {"not_present": ["$input_a"]}]})",
                                "$input_a"},
        // The unary form, a bare operand instead of an array.
        UnresolvedReferenceCase{
            "BarePresenceOfARequiredOperand", R"({"present": "$input_a"})", "$input_a"}),
    [](const ::testing::TestParamInfo<UnresolvedReferenceCase>& info) { return info.param.name; });

TEST(TestCompiledCriteria, ANativeGraphMatchChecksKernelAndDeviceReferences)
{
    // A native match publishes whatever it returns at runtime, so its tokens are unknown at
    // load. The KMD and the device vocabulary are known, and the device is read before the
    // binding, so both are checked.
    const GraphMatch native{NativeGraphMatch{"test.native"}};
    const auto schema = schemaWith(
        {{"dtype", MetadataType::STRING, std::nullopt}, {"lds", MetadataType::INT, std::nullopt}});

    EXPECT_EQ(criteriaReferenceError(
                  *criteria(R"({"==": ["$q.anything", "$kernel.dtype"]})", MatchScope::KERNEL),
                  native,
                  schema),
              "");
    EXPECT_EQ(criteriaReferenceError(
                  *criteria(R"({"<=": ["$kernel.lds", "$device.lds_size"]})", MatchScope::KERNEL),
                  native,
                  schema),
              "");
    EXPECT_NE(criteriaReferenceError(
                  *criteria(R"({"==": ["$kernel.tile", 1]})", MatchScope::KERNEL), native, schema),
              "");
    EXPECT_NE(criteriaReferenceError(
                  *criteria(R"({"<=": ["$kernel.lds", "$device.lds_sz"]})", MatchScope::KERNEL),
                  native,
                  schema)
                  .find("$device.lds_sz"),
              std::string::npos);
}

TEST(TestCompiledCriteria, AnEngineWithoutGraphMatchPublishesOnlyKernelAndDevice)
{
    const GraphMatch none{};
    const auto schema = schemaWith({{"lds", MetadataType::INT, std::nullopt}});

    EXPECT_EQ(criteriaReferenceError(
                  *criteria(R"({"<=": ["$kernel.lds", "$device.lds_size"]})", MatchScope::KERNEL),
                  none,
                  schema),
              "");
    EXPECT_NE(criteriaReferenceError(*criteria(R"({"==": ["$graph.node_count", 1]})"), none, schema)
                  .find("$graph.node_count"),
              std::string::npos);
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

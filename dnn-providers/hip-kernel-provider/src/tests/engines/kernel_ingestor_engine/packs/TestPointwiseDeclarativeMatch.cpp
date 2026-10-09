// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/CompiledGraphPattern.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPatternMatcher.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseDeclarativeMatch.cpp
 * @brief Tests the pointwise engine with a declarative `graph_match.nodes` pattern in
 *        place of its native graph match, loaded from descriptor files on disk.
 *
 * The descriptor set copies the unit pointwise set's ADD pack (KMD, UHD, UMDs, UDD and
 * native symbols) under its own engine name and ids. It differs from the shipped engine
 * only in which `graph_match` arm binds the graph. Matching needs no device, so the
 * engine runs against a fixed device identity.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildConvFwdGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildPointwiseGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildTwoNodePointwiseGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::GraphFixture;
using hip_kernel_provider::kernel_ingestor_engine::testing::matchesGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::POINTWISE_ADD;
using hip_kernel_provider::kernel_ingestor_engine::testing::testDeviceProperties;
using hip_kernel_provider::kernel_ingestor_engine::testing::transformGraph;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* SCRATCH_LABEL = "pointwisedeclarative";
constexpr const char* ENGINE_NAME = "hipkernel:PointwiseDeclarative";

constexpr const char* ENGINE_ID = "3e5b0c71-9a24-4d8f-b613-0f7e2c9a4d15";
constexpr const char* SCHEMA_ID = "6a0d4f92-1c37-4b85-9e20-d4b7f13a8c06";
constexpr const char* HEURISTIC_ID = "c27f8e40-5d19-4a63-8b0e-91a4d6c2f357";
constexpr const char* ADD_MATCHER_ID = "0b94e3d7-62a8-4f1c-a5d3-7e8c20b4f619";
constexpr const char* KERNEL_MATCHER_ID = "f4c1a8e2-3b70-4d96-8e15-a2d97c6b0e38";
constexpr const char* DISPATCH_ID = "8d27b6f5-0e43-4c18-b9a7-35f1e8d2c604";
constexpr const char* PACK_ID = "1e6f3a90-c82d-4b57-a4e1-6d0b9f2c7385";

/// One binary pointwise node. Its inputs and output bind to the roots the native match
/// publishes.
nlohmann::json pointwisePattern()
{
    return nlohmann::json::array({{{"kind", "op"},
                                   {"id", "pointwise"},
                                   {"op", "pointwise"},
                                   {"operands", {{"in_0", "$input_a"}, {"in_1", "$input_b"}}},
                                   {"results", {{"out_0", "$output"}}}}});
}

nlohmann::json embeddedAddKernel(const char* id, int64_t blockSize)
{
    return {{"version", "1.0"},
            {"id", id},
            {"name", "pointwise_add.f32_block" + std::to_string(blockSize)},
            {"kernel_source",
             {{"kind", "embedded_source"},
              {"source_file", "kernels/PointwiseAdd.cpp"},
              {"entry_point", "PointwiseAdd"}}},
            {"metadata", {{"block_size", blockSize}, {"dtype", "FLOAT"}, {"operation", "ADD"}}},
            {"priority", 0}};
}

void writeJson(const std::filesystem::path& path, const nlohmann::json& body)
{
    std::ofstream(path, std::ios::binary) << body.dump(2) << '\n';
}

/// Writes one descriptor file per type, like the unit pointwise set, with the UED's
/// `graph_match` on the declarative arm.
void writeDeclarativeSet(const std::filesystem::path& root)
{
    writeJson(
        root / "pointwise.kmd.json",
        {{"version", "1.0"},
         {"id", SCHEMA_ID},
         {"name", "pointwise variant fields"},
         {"fields",
          nlohmann::json::array({{{"name", "block_size"}, {"type", "int"}, {"default_value", 64}},
                                 {{"name", "dtype"}, {"type", "string"}},
                                 {{"name", "operation"}, {"type", "string"}}})}});
    writeJson(root / "pointwise.uhd.json",
              {{"version", "1.0"},
               {"id", HEURISTIC_ID},
               {"name", "pointwise selector"},
               {"kind", "native"},
               {"payload", std::string(POINTWISE_ADD.score)}});
    writeJson(root / "pointwise.ued.json",
              {{"version", "1.0"},
               {"id", ENGINE_ID},
               {"name", ENGINE_NAME},
               {"graph_match", {{"nodes", pointwisePattern()}}},
               {"heuristic", HEURISTIC_ID},
               {"metadata", SCHEMA_ID},
               {"knobs", nlohmann::json::array({"block_size"})},
               {"behavior_notes", nlohmann::json::array({"runtime_compilation"})}});
    writeJson(root / "operation_is_add.umd.json",
              {{"version", "1.0"},
               {"id", ADD_MATCHER_ID},
               {"name", "graph operation is add"},
               {"scope", "graph"},
               {"match_symbol", std::string(POINTWISE_ADD.operationMatcher)}});
    writeJson(root / "kernel_dtype_matches_graph.umd.json",
              {{"version", "1.0"},
               {"id", KERNEL_MATCHER_ID},
               {"name", "kernel dtype matches the graph's dtype"},
               {"scope", "kernel"},
               {"match_symbol", std::string(POINTWISE_ADD.kernelMatcher)}});
    writeJson(root / "pointwise.udd.json",
              {{"version", "1.0"},
               {"id", DISPATCH_ID},
               {"name", "pointwise dispatch"},
               {"dispatch_symbol", std::string(POINTWISE_ADD.dispatch)}});
    writeJson(root / "pointwise_add.kdp.json",
              {{"version", "1.0"},
               {"id", PACK_ID},
               {"name", "hipkernel:pointwise_add_declarative"},
               {"arch", nlohmann::json::array()},
               {"matchers", nlohmann::json::array({ADD_MATCHER_ID, KERNEL_MATCHER_ID})},
               {"engine", ENGINE_ID},
               {"dispatch", DISPATCH_ID},
               {"kernelDescriptors",
                nlohmann::json::array(
                    {embeddedAddKernel("5f2a9c14-7e63-4b08-9d1f-c4a6e0b83d72", 64),
                     embeddedAddKernel("a93d1e75-2c40-4f8b-b6e9-07d5f8c1a246", 256)})}});
}

/// Loads and validates the set the way the provider does: register native symbols, then
/// read every file from disk.
std::vector<ValidatedSet<Handle>> loadDeclarativeSets(const ScopedDirectory& scratch)
{
    registerNativeIngestorSymbols();
    writeDeclarativeSet(scratch.path());
    return loadValidatedDescriptorSets<Handle>(scratch.path());
}

/// A fixed device identity. Matching reads only the arch, so no device is needed.
class FixedDeviceResolver : public IDeviceResolver<Handle>
{
public:
    DeviceId deviceId(const Handle& /*handle*/) const override
    {
        return 0;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceProperties _properties = testDeviceProperties();
};

struct ApplicabilityCase
{
    std::string name;
    std::function<flatbuffers::FlatBufferBuilder()> build;
    bool applicable;
};

class TestPointwiseDeclarativeMatch : public ::testing::TestWithParam<ApplicabilityCase>
{
};

std::set<std::string> tokenNames(const BoundTokens& bound)
{
    std::set<std::string> names;
    for(const auto& token : bound)
    {
        names.insert(token.first);
    }
    return names;
}

/// Asserts both arms bind @p builder's graph, to the same tokens and values.
void expectNativeParity(flatbuffers::FlatBufferBuilder builder)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const auto sets = loadDeclarativeSets(scratch);
    ASSERT_EQ(sets.size(), 1U);
    const auto* pattern = std::get_if<std::shared_ptr<const CompiledGraphPattern>>(
        &sets.front().set.engine.graphMatch);
    ASSERT_NE(pattern, nullptr);
    ASSERT_NE(*pattern, nullptr);

    const GraphFixture fixture(std::move(builder));
    const auto native = matchesGraph(POINTWISE_ADD, fixture.context());
    const auto declarative = matchGraphPattern(**pattern, fixture.context());
    ASSERT_TRUE(native.has_value());
    ASSERT_TRUE(declarative.has_value());

    ASSERT_EQ(tokenNames(*declarative), tokenNames(*native));
    for(const auto& [name, value] : *native)
    {
        EXPECT_TRUE(declarative->at(name) == value) << name;
    }
}

/// A binary add with every optional pointwise attribute set. The node compute type differs
/// from the tensor dtype to catch a publication that reads the wrong one.
flatbuffers::FlatBufferBuilder buildFullyAttributedPointwiseGraph()
{
    return transformGraph(buildPointwiseGraph(), [](auto& graph) {
        graph.nodes[0]->compute_data_type = data_objects::DataType::DOUBLE;
        auto* attributes = graph.nodes[0]->attributes.AsPointwiseAttributes();
        attributes->relu_lower_clip = 0.5f;
        attributes->relu_upper_clip = 6.0f;
        attributes->relu_lower_clip_slope = 0.25f;
        attributes->axis_tensor_uid = 7;
        attributes->swish_beta = 1.5f;
        attributes->elu_alpha = 2.0f;
        attributes->softplus_beta = 3.0f;
    });
}

} // namespace

/// The pattern compiles at load, the set validates, and the engine answers applicability
/// through the compiled pattern.
TEST_P(TestPointwiseDeclarativeMatch, EngineApplicabilityFollowsThePattern)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const auto sets = loadDeclarativeSets(scratch);
    ASSERT_EQ(sets.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<const CompiledGraphPattern>>(
        sets.front().set.engine.graphMatch));

    const FixedDeviceResolver resolver;
    const auto engine = makeEngine<Handle, Settings, Context>(sets.front().set, resolver);
    const GraphFixture fixture(GetParam().build());
    Handle handle;

    EXPECT_EQ(engine->isApplicable(handle, fixture.context().graph), GetParam().applicable);
}

INSTANTIATE_TEST_SUITE_P(
    Graphs,
    TestPointwiseDeclarativeMatch,
    ::testing::Values(
        ApplicabilityCase{"BinaryAdd", [] { return buildPointwiseGraph(); }, true},
        // Full coverage: a one-node pattern matches only a one-node graph.
        ApplicabilityCase{"TwoPointwiseNodes", [] { return buildTwoNodePointwiseGraph(); }, false},
        // The pattern leaves the optional `in_2` unnamed, so it does not describe a
        // ternary op.
        ApplicabilityCase{"TernaryPointwise",
                          [] {
                              return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                                         data_objects::DataType::FLOAT,
                                                         {1, 1, 1, 1},
                                                         std::nullopt,
                                                         true,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         /*includeThirdOperand=*/true);
                          },
                          false},
        ApplicabilityCase{"ConvolutionForward", [] { return buildConvFwdGraph(); }, false},
        // The pattern admits these two, but the kernel writes one element and its layout
        // needs rank 4 or 5. The pack's criterion re-checks the binding and declines them.
        ApplicabilityCase{"MultiElementTensors",
                          [] {
                              return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                                         data_objects::DataType::FLOAT,
                                                         {1, 1, 2, 2});
                          },
                          false},
        ApplicabilityCase{"Rank3Tensors",
                          [] {
                              return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                                         data_objects::DataType::FLOAT,
                                                         {1, 1, 1});
                          },
                          false}),
    [](const ::testing::TestParamInfo<ApplicabilityCase>& info) { return info.param.name; });

/// On a graph both arms accept, the pattern binds the same tokens and values as the native
/// match, so UMDs and dispatch formulas work under either arm.
TEST(TestPointwiseDeclarativeMatchBinding, PublishesTheNativeBindingTokenForToken)
{
    expectNativeParity(buildPointwiseGraph());
}

/// Parity for every optional node attribute. The native match publishes them by hand and
/// the declarative arm uses the op-schema registry. Only this test catches drift between
/// the two.
TEST(TestPointwiseDeclarativeMatchBinding, PublishesEveryOptionalAttributeTokenForToken)
{
    // Parity over absent keys proves nothing, so confirm the native binding has them all.
    const GraphFixture fixture(buildFullyAttributedPointwiseGraph());
    const auto native = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(native.has_value());
    for(const auto* key : {"pointwise.relu_lower_clip",
                           "pointwise.relu_upper_clip",
                           "pointwise.relu_lower_clip_slope",
                           "pointwise.axis_tensor_uid",
                           "pointwise.swish_beta",
                           "pointwise.elu_alpha",
                           "pointwise.softplus_beta"})
    {
        EXPECT_EQ(native->count(key), 1U) << key;
    }

    expectNativeParity(buildFullyAttributedPointwiseGraph());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
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
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseDeclarativeMatch.cpp
 * @brief Tests the pointwise engine with a declarative `graph_match.nodes` pattern in
 *        place of its native graph match.
 *
 * The engine test copies the shipped unit pointwise set from disk and replaces only its
 * UED's `graph_match`, so the two engines differ only in which arm binds the graph.
 * Matching needs no device, so the engine runs against a fixed device identity.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildPointwiseGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::expectSameBinding;
using hip_kernel_provider::kernel_ingestor_engine::testing::GraphFixture;
using hip_kernel_provider::kernel_ingestor_engine::testing::loadedSet;
using hip_kernel_provider::kernel_ingestor_engine::testing::matchesGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::matchPattern;
using hip_kernel_provider::kernel_ingestor_engine::testing::POINTWISE_ADD;
using hip_kernel_provider::kernel_ingestor_engine::testing::pointwisePattern;
using hip_kernel_provider::kernel_ingestor_engine::testing::testDeviceProperties;
using hip_kernel_provider::kernel_ingestor_engine::testing::transformGraph;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* SCRATCH_LABEL = "pointwisedeclarative";

/// A fixed device identity. Matching reads only the arch, so no device is needed.
class FixedDeviceResolver : public IDeviceResolver<Handle>
{
public:
    explicit FixedDeviceResolver(DeviceProperties properties)
        : _properties(std::move(properties))
    {
    }

    DeviceId deviceId(const Handle& /*handle*/) const override
    {
        return 0;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceProperties _properties;
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

/// Asserts the native match and the declarative pattern both bind @p builder's graph, to
/// the same tokens and values.
void expectNativeParity(flatbuffers::FlatBufferBuilder builder)
{
    const GraphFixture fixture(std::move(builder));
    const auto native = matchesGraph(POINTWISE_ADD, fixture.context());
    const auto declarative = matchPattern(pointwisePattern(), fixture.context());
    ASSERT_TRUE(native.has_value());
    ASSERT_TRUE(declarative.has_value());

    expectSameBinding(*native,
                      *declarative,
                      {"pointwise.input_a.",
                       "pointwise.input_b.",
                       "pointwise.output.",
                       "pointwise.flops",
                       "pointwise.bytes"});
}

/// A binary add with every optional pointwise attribute set. The node compute type differs
/// from both the tensor dtype and the graph's default compute type, to catch a publication
/// that reads the wrong one.
flatbuffers::FlatBufferBuilder buildFullyAttributedPointwiseGraph()
{
    return transformGraph(buildPointwiseGraph(), [](auto& graph) {
        graph.compute_data_type = data_objects::DataType::HALF;
        graph.is_override_shape_enabled = true;
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
    // One staged shard of the shipped set: every descriptor of it sits beside its packs.
    const auto& shipped = loadedSet(POINTWISE_ADD.engineName);
    ASSERT_FALSE(shipped.packs.empty());
    const auto& pack = shipped.packs.front();
    ASSERT_FALSE(pack.kernels.empty());

    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    std::filesystem::copy(pack.kernels.front().originDirectory,
                          scratch.path(),
                          std::filesystem::copy_options::recursive);
    const auto uedPath = scratch.path() / "pointwise.ued.json";
    auto ued = nlohmann::json::parse(std::ifstream(uedPath));
    ued["graph_match"] = {{"nodes", pointwisePattern()}};
    std::ofstream(uedPath, std::ios::binary | std::ios::trunc) << ued.dump();

    registerNativeIngestorSymbols();
    auto sets = loadValidatedDescriptorSets<Handle>(scratch.path());
    ASSERT_EQ(sets.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<const CompiledGraphPattern>>(
        sets.front().set.engine.graphMatch));

    // The staged packs name the architecture they were packed for.
    auto properties = testDeviceProperties();
    if(!pack.arch.empty())
    {
        properties.gcnArchName = pack.arch.front();
    }
    const FixedDeviceResolver resolver(std::move(properties));
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

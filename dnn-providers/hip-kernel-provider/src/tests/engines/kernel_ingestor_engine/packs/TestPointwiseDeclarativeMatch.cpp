// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
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
 *        place of its native graph match, and with declarative `criteria` in place of its
 *        native UMDs.
 *
 * The engine test copies the shipped unit pointwise set from disk and replaces only its
 * UED's `graph_match`, and optionally the ADD pack's UMDs, so the engines differ only in
 * which arm binds the graph and which arm decides applicability. Matching needs no device,
 * so the engine runs against a fixed device identity.
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

/// `{name: [args...]}`.
nlohmann::json op(const char* name, std::vector<nlohmann::json> args)
{
    return nlohmann::json::object({{name, nlohmann::json(std::move(args))}});
}

/// The native pointwise criteria restated as expressions over the pattern's binding: one
/// node, three single-element rank-4 or rank-5 operands of one dtype in a supported layout,
/// none virtual, passed by value, or constant. Dims are positive, so a product of 1 means
/// every axis is 1; a rank-4 tensor has no `dims[4]`, which the default fills.
std::vector<nlohmann::json> servableBindingCriteria()
{
    std::vector<nlohmann::json> all{op("==", {"$graph.node_count", 1})};
    for(const std::string root : {"$input_a", "$input_b", "$output"})
    {
        all.push_back(op("in", {root + ".rank", nlohmann::json::array({4, 5})}));
        all.push_back(op("==",
                         {op("*",
                             {root + ".dims[0]",
                              root + ".dims[1]",
                              root + ".dims[2]",
                              root + ".dims[3]",
                              op("value_or_default", {root + ".dims[4]", 1})}),
                          1}));
        all.push_back(op(
            "in",
            {root + ".stride_order", nlohmann::json::array({"nchw", "nhwc", "ncdhw", "ndhwc"})}));
        all.push_back(op("!", {root + ".virtual"}));
        all.push_back(op("!", {root + ".is_runtime_pass_by_value"}));
        all.push_back(op("not_present", {root + ".value_f32"}));
    }
    all.push_back(op("==", {"$input_b.dtype", "$input_a.dtype"}));
    all.push_back(op("==", {"$output.dtype", "$input_a.dtype"}));
    return all;
}

/// Replaces the native symbol of the UMD in @p path with @p criteria.
void declareCriteria(const std::filesystem::path& path, nlohmann::json criteria)
{
    auto umd = nlohmann::json::parse(std::ifstream(path));
    umd.erase("match_symbol");
    umd["criteria"] = std::move(criteria);
    std::ofstream(path, std::ios::binary | std::ios::trunc) << umd.dump();
}

/// Loads one staged shard of the shipped pointwise set with its UED's `graph_match`
/// replaced by the declarative pattern and, when @p declarativeCriteria, the ADD pack's
/// UMDs replaced by servableBindingCriteria(), then asks the engine about @p graph.
void expectApplicability(const ApplicabilityCase& graph, bool declarativeCriteria)
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

    if(declarativeCriteria)
    {
        auto operation = servableBindingCriteria();
        operation.push_back(op("==", {"$pointwise.operation", "ADD"}));
        declareCriteria(scratch.path() / "operation_is_add.umd.json", op("and", operation));
        auto dtype = servableBindingCriteria();
        dtype.push_back(op("==", {"$input_a.dtype", "$kernel.dtype"}));
        declareCriteria(scratch.path() / "kernel_dtype_matches_graph.umd.json", op("and", dtype));
    }

    registerNativeIngestorSymbols();
    auto sets = loadValidatedDescriptorSets<Handle>(scratch.path());
    ASSERT_EQ(sets.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<const CompiledGraphPattern>>(
        sets.front().set.engine.graphMatch));
    if(declarativeCriteria)
    {
        const auto& matchers = sets.front().set.matchers;
        ASSERT_EQ(std::count_if(matchers.begin(),
                                matchers.end(),
                                [](const MatchDescriptor& m) { return m.criteria != nullptr; }),
                  2);
    }

    // The staged packs name the architecture they were packed for.
    auto properties = testDeviceProperties();
    if(!pack.arch.empty())
    {
        properties.gcnArchName = pack.arch.front();
    }
    const FixedDeviceResolver resolver(std::move(properties));
    const auto engine = makeEngine<Handle, Settings, Context>(sets.front().set, resolver);
    const GraphFixture fixture(graph.build());
    Handle handle;

    EXPECT_EQ(engine->isApplicable(handle, fixture.context().graph), graph.applicable);
}

} // namespace

/// The pattern compiles at load, the set validates, and the engine answers applicability
/// through the compiled pattern.
TEST_P(TestPointwiseDeclarativeMatch, EngineApplicabilityFollowsThePattern)
{
    expectApplicability(GetParam(), false);
}

/// With the ADD pack's native criteria restated as `criteria` expressions over the
/// pattern's binding, the engine answers every graph as the native criteria do.
TEST_P(TestPointwiseDeclarativeMatch, DeclarativeCriteriaAnswerAsTheNativeOnesDo)
{
    expectApplicability(GetParam(), true);
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

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx950AttentionDenseSchemaGate.cpp
 * @brief The schema gate is what keeps ragged and non-default-alignment SDPA graphs away
 *        from the shipped hipkernel:Gfx950AttentionDense engine.
 *
 * The engine's graph_match checks neither a ragged_offset on Q/K/V/O nor a declared
 * tensor alignment. Both features need engine API 1.3.0: hipDNN stamps such a graph's
 * min_required_engine_api_version 1.3.0, a plugin at API 1.4.0 passes the host's plugin
 * filter, and the remaining guard is GenericPlanBuilder::understandsGraph(), which
 * declines a graph whose stamp is above the engine descriptor's sdk_version. This suite
 * offers both graphs to an engine built from the loaded descriptor set, the way the
 * provider builds it, and requires that decline. It fails if the gate goes or the
 * descriptor's sdk_version rises to 1.3.0: raise it only once the matcher declines ragged
 * operands and a declared alignment it cannot serve itself.
 *
 * Hand-written, not generated: it belongs to the census binary because it reads the
 * shipped descriptor set, and is registered with its own hkp_register_census_tests() call
 * in src/tests/CMakeLists.txt. That shard is built only where the engine ships for
 * gfx950, so IngestorGenerator's test_gfx950_attention_dense_schema_gate.py pins the
 * configured sdk_version below 1.3.0 in every build.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace d = hipdnn_flatbuffers_sdk::data_objects;
namespace ingestor = hipdnn_plugin_sdk::ingestor;
using hipdnn_data_sdk::utilities::Version;

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t RAGGED_OFFSET_UID = 5;

constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 4;
constexpr int64_t SEQ = 256;
constexpr int64_t HEAD_SIZE = 128;

/// A device-less gfx950: if the gate let a graph through, the engine's real graph_match
/// would run against the architecture it ships for.
class Gfx950Resolver : public ingestor::IDeviceResolver<Handle>
{
public:
    ingestor::DeviceId deviceId(const Handle& /*handle*/) const override
    {
        return 0;
    }

    const ingestor::DeviceProperties&
        deviceProperties(ingestor::DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    ingestor::DeviceProperties _properties = [] {
        ingestor::DeviceProperties properties;
        properties.gcnArchName = "gfx950";
        properties.warpSize = 64;
        return properties;
    }();
};

std::unique_ptr<d::TensorAttributesT>
    tensor(int64_t uid, d::DataType type, std::vector<int64_t> dims, std::vector<int64_t> strides)
{
    auto t = std::make_unique<d::TensorAttributesT>();
    t->uid = uid;
    t->data_type = type;
    t->dims = std::move(dims);
    t->strides = std::move(strides);
    return t;
}

/// The graph Gfx950AttentionDense serves: BF16 BSHD Q/K/V/O `[2, 4, 256, 128]`, top-left
/// causal, compute FLOAT.
d::GraphT denseGraph()
{
    d::GraphT g;
    const std::vector<int64_t> dims = {BATCH, HEADS, SEQ, HEAD_SIZE};
    const std::vector<int64_t> bshd = {SEQ * HEADS * HEAD_SIZE, HEAD_SIZE, HEADS * HEAD_SIZE, 1};
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        g.tensors.push_back(tensor(uid, d::DataType::BFLOAT16, dims, bshd));
    }
    d::SdpaAttributesT a;
    a.q_tensor_uid = Q_UID;
    a.k_tensor_uid = K_UID;
    a.v_tensor_uid = V_UID;
    a.o_tensor_uid = O_UID;
    a.right_bound = 0;
    a.diagonal_alignment = d::DiagonalAlignment::TOP_LEFT;
    a.attn_scale_value = 0.08838835F;
    auto n = std::make_unique<d::NodeT>();
    n->compute_data_type = d::DataType::FLOAT;
    n->attributes.Set(a);
    g.nodes.push_back(std::move(n));
    return g;
}

/// Q and O ragged at the sequence axis through one INT32 `[B+1, 1, 1, 1]` offset tensor.
d::GraphT raggedGraph()
{
    auto g = denseGraph();
    for(auto& t : g.tensors)
    {
        if(t->uid == Q_UID || t->uid == O_UID)
        {
            t->ragged_offset_tensor_uid = RAGGED_OFFSET_UID;
        }
    }
    g.tensors.push_back(
        tensor(RAGGED_OFFSET_UID, d::DataType::INT32, {BATCH + 1, 1, 1, 1}, {1, 1, 1, 1}));
    return g;
}

/// Q declaring an 8-byte device-pointer alignment instead of the default 16.
d::GraphT eightByteAlignedGraph()
{
    auto g = denseGraph();
    g.tensors[0]->alignment = 8;
    return g;
}

/// Stamps @p g as hipDNN's GraphDescriptor does: the floor from the shared
/// graph -> version mapping. Returns that floor.
Version stamp(d::GraphT& g, bool ragged, bool nonDefaultAlignment)
{
    const auto& floor = hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(
        false, false, ragged, nonDefaultAlignment, false);
    g.min_required_engine_api_version
        = std::make_unique<d::EngineApiVersion>(hipdnn_plugin_sdk::toEngineApiVersion(floor));
    return floor;
}

/// Offers @p g to an engine built from the loaded Gfx950AttentionDense set, as the
/// provider builds it, and returns whether the engine found it applicable.
bool offer(const d::GraphT& g)
{
    const Gfx950Resolver resolver;
    const auto engine = ingestor::makeEngine<Handle, Settings, Context>(
        loadedSet(GFX950_ATTENTION_DENSE_ENGINE_NAME), resolver);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(d::Graph::Pack(builder, &g));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        builder.GetBufferPointer(), builder.GetSize());
    Handle handle;
    return engine->isApplicable(handle, wrapper);
}

/// The decline GenericPlanBuilder::understandsGraph() logs for @p floor.
std::string schemaGateDecline(const Version& floor)
{
    const auto& set = loadedSet(GFX950_ATTENTION_DENSE_ENGINE_NAME);
    return "ingestor: engine '" + std::string(GFX950_ATTENTION_DENSE_ENGINE_NAME)
           + "' declined the graph: it understands graph schema " + set.engine.sdkVersion.str()
           + " but this graph requires " + floor.str();
}

/// hip_kernel_provider's advertised engine plugin API version, which the host compares
/// with the stamp before any engine sees the graph.
Version pluginApiVersion()
{
    const char* version = nullptr;
    EXPECT_EQ(hipdnnPluginGetApiVersion(&version), HIPDNN_PLUGIN_STATUS_SUCCESS);
    return Version{std::string(version != nullptr ? version : "0.0.0")};
}

TEST(TestGfx950AttentionDenseSchemaGate, DeclinesARaggedGraphAtTheSchemaGate)
{
    auto g = raggedGraph();
    const auto floor = stamp(g, true, false);
    ASSERT_EQ(floor.str(), "1.3.0");
    // The plugin passes the host's filter, so only the engine's gate is left.
    ASSERT_TRUE(pluginApiVersion() >= floor) << pluginApiVersion().str();

    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_FALSE(offer(g));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, schemaGateDecline(floor)))
        << "expected: " << schemaGateDecline(floor);
}

TEST(TestGfx950AttentionDenseSchemaGate, DeclinesANonDefaultAlignmentGraphAtTheSchemaGate)
{
    auto g = eightByteAlignedGraph();
    const auto floor = stamp(g, false, true);
    ASSERT_EQ(floor.str(), "1.3.0");
    ASSERT_TRUE(pluginApiVersion() >= floor) << pluginApiVersion().str();

    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_FALSE(offer(g));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, schemaGateDecline(floor)))
        << "expected: " << schemaGateDecline(floor);
}

/// Control: the same graph without either feature is stamped at the baseline, passes the
/// gate and is served, so the declines above come from the stamp alone.
TEST(TestGfx950AttentionDenseSchemaGate, PassesTheBaselineGraphThroughTheSchemaGate)
{
    auto g = denseGraph();
    const auto floor = stamp(g, false, false);
    ASSERT_EQ(floor.str(), "1.0.0");

    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_TRUE(offer(g));
    EXPECT_FALSE(recorder.hasLogContaining("understands graph schema"));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

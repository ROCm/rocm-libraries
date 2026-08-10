// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <stdexcept>

/**
 * @file IntegrationGpuKernelIngestor.cpp
 * @brief The kernel ingestor, end to end across the plugin ABI.
 *
 * Loads the real built plugin and drives every call hipDNN makes, in the order it makes
 * them: applicability, engine details, workspace, execution context, execute. Nothing
 * here links the provider's internals — the only surface used is the C ABI a host would
 * use — so this is what proves the descriptor set, the matchers, the heuristic, the
 * dispatch handler, and the engine registration actually compose.
 *
 * The unit tests cover each of those pieces in isolation. What they cannot show is that
 * the pieces agree with one another through the ABI, which is the whole claim of a
 * data-driven ingestor.
 */
namespace hip_kernel_provider::ingestor_poc::integration
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr int64_t INPUT_A_UID = 1;
constexpr int64_t INPUT_B_UID = 2;
constexpr int64_t OUTPUT_UID = 3;
constexpr const char* ENGINE_NAME = "kernel_ingestor_poc:PointwiseAdd";
constexpr const char* BLOCK_SIZE_KNOB = "block_size";

/// Workspace the pack's larger-block kernel declares, which the engine reports as the
/// maximum across its surviving kernels.
constexpr size_t EXPECTED_WORKSPACE_BYTES = 1024;

/// A serialized single-node pointwise-ADD graph over 1-element 4-D FLOAT tensors: the
/// one shape this POC's pack accepts.
flatbuffers::DetachedBuffer buildPointwiseAddGraph(bool withIdentity = true)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const std::vector<int64_t> strides = {1, 1, 1, 1};

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const auto uid : {INPUT_A_UID, INPUT_B_UID, OUTPUT_UID})
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, uid, nullptr, data_objects::DataType::FLOAT, &strides, &dims, false));
    }

    data_objects::PointwiseAttributesBuilder attributesBuilder(builder);
    attributesBuilder.add_operation(data_objects::PointwiseMode::ADD);
    attributesBuilder.add_in_0_tensor_uid(INPUT_A_UID);
    attributesBuilder.add_in_1_tensor_uid(INPUT_B_UID);
    attributesBuilder.add_out_0_tensor_uid(OUTPUT_UID);
    auto attributes = attributesBuilder.Finish();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "pointwise_add",
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::PointwiseAttributes,
                                       attributes.Union()));

    auto name = builder.CreateString("ingestor_poc_pointwise_add");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    // A finalized graph carries an identity, which is what the provider keys its
    // catalog cache on. A graph without one still matches, just uncached.
    hipdnn_flatbuffers_sdk::utilities::UuidBytes idBytes{};
    idBytes.fill(0x42);
    const auto uuid = hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(idBytes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    if(withIdentity)
    {
        graphBuilder.add_id(&uuid);
    }
    builder.Finish(graphBuilder.Finish());

    return builder.Release();
}

/// A graph this pack must decline: two nodes, so no single prebuilt kernel serves it.
flatbuffers::DetachedBuffer buildUnsupportedGraph()
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const std::vector<int64_t> strides = {1, 1, 1, 1};
    constexpr int64_t INTERMEDIATE_UID = 4;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const auto uid : {INPUT_A_UID, INPUT_B_UID, OUTPUT_UID, INTERMEDIATE_UID})
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     uid,
                                                                     nullptr,
                                                                     data_objects::DataType::FLOAT,
                                                                     &strides,
                                                                     &dims,
                                                                     uid == INTERMEDIATE_UID));
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(const auto& uids :
        std::vector<std::array<int64_t, 3>>{{INPUT_A_UID, INPUT_B_UID, INTERMEDIATE_UID},
                                            {INTERMEDIATE_UID, INPUT_B_UID, OUTPUT_UID}})
    {
        data_objects::PointwiseAttributesBuilder attributesBuilder(builder);
        attributesBuilder.add_operation(data_objects::PointwiseMode::ADD);
        attributesBuilder.add_in_0_tensor_uid(uids[0]);
        attributesBuilder.add_in_1_tensor_uid(uids[1]);
        attributesBuilder.add_out_0_tensor_uid(uids[2]);
        auto attributes = attributesBuilder.Finish();

        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "pointwise_add",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           attributes.Union()));
    }

    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "two_node_pointwise",
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::DataType::FLOAT,
                                                   &tensors,
                                                   &nodes));

    return builder.Release();
}

/// An engine config naming this engine, with no knob settings.
flatbuffers::DetachedBuffer buildEngineConfig(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    data_objects::EngineConfigBuilder configBuilder(builder);
    configBuilder.add_engine_id(engineId);
    builder.Finish(configBuilder.Finish());
    return builder.Release();
}

hipdnnPluginConstData_t asConstData(const flatbuffers::DetachedBuffer& buffer)
{
    return {buffer.data(), buffer.size()};
}

/// Device buffers for one 1-element add, freed on scope exit.
class AddBuffers
{
public:
    AddBuffers(float a, float b)
    {
        EXPECT_EQ(hipMalloc(&_a, sizeof(float)), hipSuccess);
        EXPECT_EQ(hipMalloc(&_b, sizeof(float)), hipSuccess);
        EXPECT_EQ(hipMalloc(&_c, sizeof(float)), hipSuccess);
        EXPECT_EQ(hipMemcpy(_a, &a, sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        EXPECT_EQ(hipMemcpy(_b, &b, sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    }

    ~AddBuffers()
    {
        static_cast<void>(hipFree(_a));
        static_cast<void>(hipFree(_b));
        static_cast<void>(hipFree(_c));
    }

    AddBuffers(const AddBuffers&) = delete;
    AddBuffers& operator=(const AddBuffers&) = delete;

    std::array<hipdnnPluginDeviceBuffer_t, 3> descriptors() const
    {
        return {hipdnnPluginDeviceBuffer_t{INPUT_A_UID, _a},
                hipdnnPluginDeviceBuffer_t{INPUT_B_UID, _b},
                hipdnnPluginDeviceBuffer_t{OUTPUT_UID, _c}};
    }

    float readResult() const
    {
        float result = 0.0f;
        EXPECT_EQ(hipMemcpy(&result, _c, sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
        return result;
    }

private:
    void* _a = nullptr;
    void* _b = nullptr;
    void* _c = nullptr;
};

/**
 * @brief The plugin's exported C API, resolved from the built shared object.
 *
 * The provider exports these symbols for hipDNN to load at run time; nothing links
 * against them. Resolving them the same way is what makes this test cross the real ABI
 * boundary rather than reaching into the provider's internals.
 */
class PluginApi
{
public:
    PluginApi()
    {
        // PLUGIN_PATH names the CMake target, so the platform's library prefix and
        // extension have to be applied to reach the file on disk.
        const std::filesystem::path pluginTarget(PLUGIN_PATH);
        const auto pluginFile = hipdnn_data_sdk::utilities::LIB_PREFIX
                                + pluginTarget.filename().string()
                                + hipdnn_data_sdk::utilities::SHARED_LIB_EXT;
        const auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / pluginTarget.parent_path()
            / pluginFile);
        _library = hipdnn_data_sdk::utilities::openLibrary(pluginPath);

        create = resolve<decltype(&hipdnnEnginePluginCreate)>("hipdnnEnginePluginCreate");
        destroy = resolve<decltype(&hipdnnEnginePluginDestroy)>("hipdnnEnginePluginDestroy");
        setStream = resolve<decltype(&hipdnnEnginePluginSetStream)>("hipdnnEnginePluginSetStream");
        getAllEngineIds = resolve<decltype(&hipdnnEnginePluginGetAllEngineIds)>(
            "hipdnnEnginePluginGetAllEngineIds");
        getApplicableEngineIds = resolve<decltype(&hipdnnEnginePluginGetApplicableEngineIds)>(
            "hipdnnEnginePluginGetApplicableEngineIds");
        getEngineDetails = resolve<decltype(&hipdnnEnginePluginGetEngineDetails)>(
            "hipdnnEnginePluginGetEngineDetails");
        destroyEngineDetails = resolve<decltype(&hipdnnEnginePluginDestroyEngineDetails)>(
            "hipdnnEnginePluginDestroyEngineDetails");
        getWorkspaceSize = resolve<decltype(&hipdnnEnginePluginGetWorkspaceSize)>(
            "hipdnnEnginePluginGetWorkspaceSize");
        createExecutionContext = resolve<decltype(&hipdnnEnginePluginCreateExecutionContext)>(
            "hipdnnEnginePluginCreateExecutionContext");
        destroyExecutionContext = resolve<decltype(&hipdnnEnginePluginDestroyExecutionContext)>(
            "hipdnnEnginePluginDestroyExecutionContext");
        getWorkspaceSizeFromExecutionContext
            = resolve<decltype(&hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext)>(
                "hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext");
        executeOpGraph = resolve<decltype(&hipdnnEnginePluginExecuteOpGraph)>(
            "hipdnnEnginePluginExecuteOpGraph");
    }

    decltype(&hipdnnEnginePluginCreate) create = nullptr;
    decltype(&hipdnnEnginePluginDestroy) destroy = nullptr;
    decltype(&hipdnnEnginePluginSetStream) setStream = nullptr;
    decltype(&hipdnnEnginePluginGetAllEngineIds) getAllEngineIds = nullptr;
    decltype(&hipdnnEnginePluginGetApplicableEngineIds) getApplicableEngineIds = nullptr;
    decltype(&hipdnnEnginePluginGetEngineDetails) getEngineDetails = nullptr;
    decltype(&hipdnnEnginePluginDestroyEngineDetails) destroyEngineDetails = nullptr;
    decltype(&hipdnnEnginePluginGetWorkspaceSize) getWorkspaceSize = nullptr;
    decltype(&hipdnnEnginePluginCreateExecutionContext) createExecutionContext = nullptr;
    decltype(&hipdnnEnginePluginDestroyExecutionContext) destroyExecutionContext = nullptr;
    decltype(&hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext)
        getWorkspaceSizeFromExecutionContext
        = nullptr;
    decltype(&hipdnnEnginePluginExecuteOpGraph) executeOpGraph = nullptr;

private:
    template <typename T>
    T resolve(const char* name) const
    {
        auto* symbol = hipdnn_data_sdk::utilities::getSymbol(_library, name);
        if(symbol == nullptr)
        {
            throw std::runtime_error("plugin does not export " + std::string(name));
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        return reinterpret_cast<T>(symbol);
    }

    hipdnn_data_sdk::utilities::SharedLibraryHandle _library = nullptr;
};

/// One process-wide load, since a plugin is loaded once and shared by every handle.
const PluginApi& pluginApi()
{
    static const PluginApi s_api;
    return s_api;
}

class IntegrationGpuKernelIngestor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(pluginApi().create(&_handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(pluginApi().setStream(_handle, _stream), HIPDNN_PLUGIN_STATUS_SUCCESS);

        _engineId = hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(pluginApi().destroy(_handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    std::vector<int64_t> applicableEngines(const hipdnnPluginConstData_t& graph) const
    {
        uint32_t count = 0;
        EXPECT_EQ(pluginApi().getApplicableEngineIds(_handle, &graph, nullptr, 0, &count),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);

        std::vector<int64_t> engines(count);
        if(count > 0)
        {
            EXPECT_EQ(
                pluginApi().getApplicableEngineIds(_handle, &graph, engines.data(), count, &count),
                HIPDNN_PLUGIN_STATUS_SUCCESS);
        }
        return engines;
    }

    bool isApplicable(const hipdnnPluginConstData_t& graph) const
    {
        const auto engines = applicableEngines(graph);
        return std::find(engines.begin(), engines.end(), _engineId) != engines.end();
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int64_t _engineId = 0;
};

// ---------------------------------------------------------------------------
// Applicability
// ---------------------------------------------------------------------------

TEST_F(IntegrationGpuKernelIngestor, ExposesTheDescriptorBackedEngineToTheHost)
{
    // The engine's id is its descriptor name hashed into hipDNN's id space, registered
    // when the engine was constructed rather than by a compile-time macro.
    uint32_t count = 0;
    ASSERT_EQ(pluginApi().getAllEngineIds(nullptr, 0, &count), HIPDNN_PLUGIN_STATUS_SUCCESS);

    std::vector<int64_t> engines(count);
    ASSERT_EQ(pluginApi().getAllEngineIds(engines.data(), count, &count),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    EXPECT_NE(std::find(engines.begin(), engines.end(), _engineId), engines.end());
}

TEST_F(IntegrationGpuKernelIngestor, AcceptsTheGraphItsDescriptorsDescribe)
{
    const auto graph = buildPointwiseAddGraph();

    EXPECT_TRUE(isApplicable(asConstData(graph)));
}

TEST_F(IntegrationGpuKernelIngestor, DeclinesAGraphNoKernelInTheCatalogServes)
{
    // Declining is free: an empty catalog answers false and hipDNN moves on. Getting
    // this wrong is what turns a cheap decline into a failed plan build.
    const auto graph = buildUnsupportedGraph();

    EXPECT_FALSE(isApplicable(asConstData(graph)));
}

TEST_F(IntegrationGpuKernelIngestor, AcceptsAGraphCarryingNoIdentity)
{
    // Without a graph id the catalog cannot be cached, but matching is unaffected: the
    // answer must be the same, only recomputed.
    const auto graph = buildPointwiseAddGraph(/*withIdentity=*/false);

    EXPECT_TRUE(isApplicable(asConstData(graph)));
}

// ---------------------------------------------------------------------------
// Engine details: knobs from the catalog
// ---------------------------------------------------------------------------

TEST_F(IntegrationGpuKernelIngestor, ReportsAKnobWhoseValuesComeFromTheCatalog)
{
    const auto graph = buildPointwiseAddGraph();
    const auto graphData = asConstData(graph);

    hipdnnPluginConstData_t details{};
    ASSERT_EQ(pluginApi().getEngineDetails(_handle, _engineId, &graphData, &details),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    const auto* engineDetails = flatbuffers::GetRoot<data_objects::EngineDetails>(details.ptr);
    ASSERT_NE(engineDetails, nullptr);
    ASSERT_NE(engineDetails->knobs(), nullptr);
    ASSERT_EQ(engineDetails->knobs()->size(), 1U);

    const auto* knob = engineDetails->knobs()->Get(0);
    ASSERT_NE(knob->knob_id(), nullptr);
    EXPECT_EQ(knob->knob_id()->str(), BLOCK_SIZE_KNOB);

    // The pack ships three kernels, but the HALF one is pruned for this FLOAT graph, so
    // the knob offers exactly the two block sizes the surviving kernels implement --
    // never the schema's theoretical range.
    const auto* constraint = knob->constraint_as_IntConstraint();
    ASSERT_NE(constraint, nullptr);
    ASSERT_NE(constraint->valid_values(), nullptr);
    std::vector<int64_t> values(constraint->valid_values()->begin(),
                                constraint->valid_values()->end());
    std::sort(values.begin(), values.end());
    EXPECT_EQ(values, (std::vector<int64_t>{64, 256}));

    // The default is whatever the heuristic ranked first, so leaving the knob alone
    // reproduces the out-of-the-box selection.
    const auto* defaultValue = knob->default_value_as_IntValue();
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(defaultValue->value(), 256);

    EXPECT_EQ(pluginApi().destroyEngineDetails(_handle, &details), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

TEST_F(IntegrationGpuKernelIngestor, ReportsTheMaximumWorkspaceAcrossSurvivingKernels)
{
    const auto graph = buildPointwiseAddGraph();
    const auto config = buildEngineConfig(_engineId);
    const auto graphData = asConstData(graph);
    const auto configData = asConstData(config);

    size_t workspaceSize = 0;
    ASSERT_EQ(pluginApi().getWorkspaceSize(_handle, &configData, &graphData, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    // One surviving kernel declares 0 bytes and the other 1024, so this answer proves
    // the query aggregates across the catalog rather than reporting one kernel's value.
    EXPECT_EQ(workspaceSize, EXPECTED_WORKSPACE_BYTES);
}

// ---------------------------------------------------------------------------
// Plan build and execute
// ---------------------------------------------------------------------------

TEST_F(IntegrationGpuKernelIngestor, ExecutesTheSelectedKernelOnDevice)
{
    const auto graph = buildPointwiseAddGraph();
    const auto config = buildEngineConfig(_engineId);
    const auto graphData = asConstData(graph);
    const auto configData = asConstData(config);

    hipdnnEnginePluginExecutionContext_t context = nullptr;
    ASSERT_EQ(pluginApi().createExecutionContext(_handle, &configData, &graphData, &context),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(context, nullptr);

    size_t workspaceSize = 0;
    ASSERT_EQ(pluginApi().getWorkspaceSizeFromExecutionContext(_handle, context, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    void* workspace = nullptr;
    if(workspaceSize > 0)
    {
        ASSERT_EQ(hipMalloc(&workspace, workspaceSize), hipSuccess);
    }

    const AddBuffers buffers(3.0f, 4.0f);
    const auto descriptors = buffers.descriptors();

    ASSERT_EQ(pluginApi().executeOpGraph(
                  _handle, context, workspace, descriptors.data(), descriptors.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

    // The whole chain, end to end: the matchers admitted this graph, the heuristic
    // ranked the catalog, the dispatch descriptor's handler compiled and launched the
    // winner, and its arguments were resolved by tensor uid.
    EXPECT_FLOAT_EQ(buffers.readResult(), 7.0f);

    EXPECT_EQ(pluginApi().destroyExecutionContext(_handle, context), HIPDNN_PLUGIN_STATUS_SUCCESS);
    if(workspace != nullptr)
    {
        EXPECT_EQ(hipFree(workspace), hipSuccess);
    }
}

TEST_F(IntegrationGpuKernelIngestor, ReusesOnePlanAcrossExecutions)
{
    const auto graph = buildPointwiseAddGraph();
    const auto config = buildEngineConfig(_engineId);
    const auto graphData = asConstData(graph);
    const auto configData = asConstData(config);

    hipdnnEnginePluginExecutionContext_t context = nullptr;
    ASSERT_EQ(pluginApi().createExecutionContext(_handle, &configData, &graphData, &context),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    size_t workspaceSize = 0;
    ASSERT_EQ(pluginApi().getWorkspaceSizeFromExecutionContext(_handle, context, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    void* workspace = nullptr;
    if(workspaceSize > 0)
    {
        ASSERT_EQ(hipMalloc(&workspace, workspaceSize), hipSuccess);
    }

    // A plan is built once and executed many times with different buffers; every
    // decision was made at build, so execution must not depend on the previous one.
    for(const auto& values : std::vector<std::array<float, 3>>{
            {1.0f, 2.0f, 3.0f}, {-5.0f, 2.5f, -2.5f}, {0.125f, 0.375f, 0.5f}})
    {
        const AddBuffers buffers(values[0], values[1]);
        const auto descriptors = buffers.descriptors();

        ASSERT_EQ(pluginApi().executeOpGraph(
                      _handle, context, workspace, descriptors.data(), descriptors.size()),
                  HIPDNN_PLUGIN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

        EXPECT_FLOAT_EQ(buffers.readResult(), values[2]);
    }

    EXPECT_EQ(pluginApi().destroyExecutionContext(_handle, context), HIPDNN_PLUGIN_STATUS_SUCCESS);
    if(workspace != nullptr)
    {
        EXPECT_EQ(hipFree(workspace), hipSuccess);
    }
}

TEST_F(IntegrationGpuKernelIngestor, ExecutesAGraphThatCannotBeCached)
{
    // Same result through the uncached path: a graph with no identity is rematched on
    // every query, which costs time and never correctness.
    const auto graph = buildPointwiseAddGraph(/*withIdentity=*/false);
    const auto config = buildEngineConfig(_engineId);
    const auto graphData = asConstData(graph);
    const auto configData = asConstData(config);

    hipdnnEnginePluginExecutionContext_t context = nullptr;
    ASSERT_EQ(pluginApi().createExecutionContext(_handle, &configData, &graphData, &context),
              HIPDNN_PLUGIN_STATUS_SUCCESS);

    size_t workspaceSize = 0;
    ASSERT_EQ(pluginApi().getWorkspaceSizeFromExecutionContext(_handle, context, &workspaceSize),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    void* workspace = nullptr;
    if(workspaceSize > 0)
    {
        ASSERT_EQ(hipMalloc(&workspace, workspaceSize), hipSuccess);
    }

    const AddBuffers buffers(2.5f, 6.5f);
    const auto descriptors = buffers.descriptors();

    ASSERT_EQ(pluginApi().executeOpGraph(
                  _handle, context, workspace, descriptors.data(), descriptors.size()),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);

    EXPECT_FLOAT_EQ(buffers.readResult(), 9.0f);

    EXPECT_EQ(pluginApi().destroyExecutionContext(_handle, context), HIPDNN_PLUGIN_STATUS_SUCCESS);
    if(workspace != nullptr)
    {
        EXPECT_EQ(hipFree(workspace), hipSuccess);
    }
}

} // namespace

} // namespace hip_kernel_provider::ingestor_poc::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

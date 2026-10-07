// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#define HIPDNN_PLUGIN_STATIC_DEFINE

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

/**
 * @file TestEnginePluginKnobCatalogApi.cpp
 * @brief Drives the C entry points EnginePluginImpl.inl generates for the optional knob
 *        catalog pair through a minimal plugin built from the SDK macros, as a provider
 *        using them would be.
 */
namespace knob_catalog_api_fixture
{

constexpr int64_t K_CATALOG_ENGINE_ID = 1;
constexpr int64_t K_PLAIN_ENGINE_ID = 2;
constexpr size_t K_CATALOG_BYTES = 16;

struct FixtureSettings
{
};

struct FixtureHandle;

struct FixtureContext : HipdnnEnginePluginExecutionContext,
                        hipdnn_plugin_sdk::ExecutionContextBase<FixtureHandle, FixtureSettings>
{
};

class FixtureContainer;

using FixtureEngineManager
    = hipdnn_plugin_sdk::EngineManager<FixtureHandle, FixtureSettings, FixtureContext>;

struct FixtureHandle : HipdnnEnginePluginHandle
{
    void setStream(hipStream_t stream)
    {
        _stream = stream;
    }

    hipStream_t getStream() const
    {
        return _stream;
    }

    std::shared_ptr<FixtureContainer> container;

    FixtureEngineManager& getEngineManager() const;

    void storeEngineDetailsDetachedBuffer(const void* ptr,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        _buffers[ptr] = std::move(buffer);
    }

    void removeEngineDetailsDetachedBuffer(const void* ptr)
    {
        _buffers.erase(ptr);
    }

    size_t bufferCount() const
    {
        return _buffers.size();
    }

private:
    hipStream_t _stream = nullptr;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>> _buffers;
};

class FixturePlan : public hipdnn_plugin_sdk::IPlan<FixtureHandle>
{
public:
    size_t getWorkspaceSize(const FixtureHandle& /*handle*/) const override
    {
        return 0;
    }

    void execute(const FixtureHandle& /*handle*/,
                 const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                 uint32_t /*numDeviceBuffers*/,
                 void* /*workspace*/) const override
    {
    }
};

/// Implements the base interface only, so its catalog is the default.
class PlainEngine
    : public hipdnn_plugin_sdk::IEngine<FixtureHandle, FixtureSettings, FixtureContext>
{
public:
    int64_t id() const override
    {
        return K_PLAIN_ENGINE_ID;
    }

    bool isApplicable(
        FixtureHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const override
    {
        return false;
    }

    void getDetails(FixtureHandle& /*handle*/,
                    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
                    hipdnnPluginConstData_t& /*detailsOut*/) const override
    {
    }

    size_t getMaxWorkspaceSize(
        const FixtureHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/)
        const override
    {
        return 0;
    }

    void initializeExecutionContext(
        const FixtureHandle& /*handle*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
        FixtureContext& executionContext) const override
    {
        executionContext.setPlan(std::make_unique<FixturePlan>());
    }
};

/// Serves a catalog into a buffer the handle owns, the way GenericEngine does.
class CatalogEngine : public PlainEngine
{
public:
    int64_t id() const override
    {
        return K_CATALOG_ENGINE_ID;
    }

    void getKnobCatalog(FixtureHandle& handle,
                        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
                        hipdnnPluginConstData_t& catalogOut) const override
    {
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(builder.CreateString(std::string(K_CATALOG_BYTES, 'c')));
        auto buffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
        catalogOut.ptr = buffer->data();
        catalogOut.size = buffer->size();
        handle.storeEngineDetailsDetachedBuffer(catalogOut.ptr, std::move(buffer));
    }
};

class FixtureContainer
{
public:
    FixtureContainer()
    {
        _engineManager.addEngine(std::make_unique<CatalogEngine>());
        _engineManager.addEngine(std::make_unique<PlainEngine>());
    }

    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines)
    {
        numEngines = 0;
        if(maxEngines > 0 && engineIds != nullptr)
        {
            engineIds[0] = K_CATALOG_ENGINE_ID;
            numEngines = 1;
        }
        return 2;
    }

    FixtureEngineManager& getEngineManager()
    {
        return _engineManager;
    }

private:
    FixtureEngineManager _engineManager;
};

inline FixtureEngineManager& FixtureHandle::getEngineManager() const
{
    return container->getEngineManager();
}

} // namespace knob_catalog_api_fixture

using namespace knob_catalog_api_fixture;

#define HIPDNN_PLUGIN_NAME "knob_catalog_api_fixture_plugin"
#define HIPDNN_PLUGIN_VERSION "1.0.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE FixtureContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE FixtureHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE FixtureContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>

namespace
{

class TestEnginePluginKnobCatalogApi : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hipdnnEnginePluginCreate(&_handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        EXPECT_EQ(hipdnnEnginePluginDestroy(_handle), HIPDNN_PLUGIN_STATUS_SUCCESS);
    }

    size_t heldBuffers() const
    {
        return static_cast<FixtureHandle*>(_handle)->bufferCount();
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
    // Not a valid Graph: the wrapper treats an unverifiable buffer as an empty graph and
    // these fixture engines never read it.
    char _graphBytes = 0;
    hipdnnPluginConstData_t _graph{&_graphBytes, 0};
};

TEST_F(TestEnginePluginKnobCatalogApi, ServedCatalogIsHandedBackAndReleasedByDestroy)
{
    hipdnnPluginConstData_t catalog{};

    ASSERT_EQ(
        hipdnnEnginePluginGetEngineKnobCatalog(_handle, K_CATALOG_ENGINE_ID, &_graph, &catalog),
        HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(catalog.ptr, nullptr);
    EXPECT_GT(catalog.size, K_CATALOG_BYTES);
    EXPECT_EQ(heldBuffers(), 1U);

    EXPECT_EQ(hipdnnEnginePluginDestroyEngineKnobCatalog(_handle, &catalog),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(heldBuffers(), 0U);
    EXPECT_EQ(catalog.ptr, nullptr);
    EXPECT_EQ(catalog.size, 0U);
}

TEST_F(TestEnginePluginKnobCatalogApi, EngineWithoutACatalogAnswersNotApplicableAndLeavesNoBuffer)
{
    hipdnnPluginConstData_t catalog{};

    EXPECT_EQ(hipdnnEnginePluginGetEngineKnobCatalog(_handle, K_PLAIN_ENGINE_ID, &_graph, &catalog),
              HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE);
    EXPECT_EQ(catalog.ptr, nullptr);
    EXPECT_EQ(heldBuffers(), 0U);
}

TEST_F(TestEnginePluginKnobCatalogApi, DecliningDoesNotOverwriteTheLastErrorOfAnEarlierFailure)
{
    // A genuine failure first, so the last-error string holds something to be clobbered.
    hipdnnPluginConstData_t catalog{};
    ASSERT_EQ(hipdnnEnginePluginGetEngineKnobCatalog(_handle, K_PLAIN_ENGINE_ID, nullptr, &catalog),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    const std::string earlier = hipdnn_plugin_sdk::PluginLastErrorManager::getLastError();
    ASSERT_FALSE(earlier.empty());

    ASSERT_EQ(hipdnnEnginePluginGetEngineKnobCatalog(_handle, K_PLAIN_ENGINE_ID, &_graph, &catalog),
              HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE);

    EXPECT_EQ(std::string(hipdnn_plugin_sdk::PluginLastErrorManager::getLastError()), earlier);
}

TEST_F(TestEnginePluginKnobCatalogApi, UnknownEngineIsAnInvalidValueNotADecline)
{
    hipdnnPluginConstData_t catalog{};

    EXPECT_EQ(hipdnnEnginePluginGetEngineKnobCatalog(_handle, 999, &_graph, &catalog),
              HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
}

TEST_F(TestEnginePluginKnobCatalogApi, NullArgumentsAreBadParams)
{
    hipdnnPluginConstData_t catalog{};

    EXPECT_EQ(
        hipdnnEnginePluginGetEngineKnobCatalog(nullptr, K_CATALOG_ENGINE_ID, &_graph, &catalog),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(
        hipdnnEnginePluginGetEngineKnobCatalog(_handle, K_CATALOG_ENGINE_ID, nullptr, &catalog),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(
        hipdnnEnginePluginGetEngineKnobCatalog(_handle, K_CATALOG_ENGINE_ID, &_graph, nullptr),
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnEnginePluginDestroyEngineKnobCatalog(_handle, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(hipdnnEnginePluginDestroyEngineKnobCatalog(_handle, &catalog),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

} // namespace

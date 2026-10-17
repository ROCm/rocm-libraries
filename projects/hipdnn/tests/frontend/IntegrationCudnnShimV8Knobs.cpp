// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// cuDNN v8 KNOB_INFO projection and KNOB_CHOICE delivery against the real hipDNN
// backend. Engine queries need hipDNN heuristics, which query the device, so these
// tests skip on device-less hosts.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_compatibility/cudnn/cudnn_graph.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_frontend/PluginPaths.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <test_plugins/TestPluginKnobRecorder.hpp>

#include "CudnnShimV8GraphBuilder.hpp"

namespace
{
using hipdnn_shim_v8_test::GraphBuilder;
using hipdnn_shim_v8_test::lastError;

class IntegrationCudnnShimV8KnobsBase : public hipdnn_tests::IntegrationTestFixture
{
protected:
    void SetUp() override
    {
        IntegrationTestFixture::SetUp();
        if(IsSkipped())
        {
            return;
        }
        _graph = _builder->reluGraph(_handle, 1, 2, {2, 4, 8, 8});
    }

    // Plans own plugin execution contexts, so they go before the handle that loaded the plugin.
    void TearDown() override
    {
        _builder.reset();
        IntegrationTestFixture::TearDown();
    }

    static std::int64_t knobInfoCount(cudnnBackendDescriptor_t engine)
    {
        std::int64_t count = -1;
        EXPECT_EQ(cudnnBackendGetAttribute(engine,
                                           CUDNN_ATTR_ENGINE_KNOB_INFO,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           0,
                                           &count,
                                           nullptr),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        return count;
    }

    cudnnStatus_t finalizeConfigWith(std::int64_t engineIndex,
                                     const std::vector<cudnnBackendDescriptor_t>& choices)
    {
        cudnnBackendDescriptor_t config
            = _builder->engineConfig(_builder->engine(_graph, engineIndex), choices);
        return cudnnBackendFinalize(config);
    }

    std::unique_ptr<GraphBuilder> _builder = std::make_unique<GraphBuilder>();
    cudnnBackendDescriptor_t _graph = nullptr;
};

// The shim's index orders engines by signed id: the context-failing engine (-32)
// is index 0 and the knob engine (-31) index 1.
constexpr std::int64_t K_CONTEXT_FAILS_INDEX = 0;
constexpr std::int64_t K_KNOB_ENGINE_INDEX = 1;

class IntegrationCudnnShimV8Knobs : public IntegrationCudnnShimV8KnobsBase
{
protected:
    void SetUp() override
    {
        IntegrationCudnnShimV8KnobsBase::SetUp();
        if(IsSkipped())
        {
            return;
        }

        // Open the exact library the backend loaded so the recorder reads its state.
        std::vector<std::filesystem::path> loadedPaths;
        const auto pathResult = hipdnn_frontend::getLoadedEnginePluginPaths(_handle, loadedPaths);
        ASSERT_TRUE(pathResult.is_good()) << pathResult.get_message();
        const auto it = std::find_if(loadedPaths.begin(), loadedPaths.end(), [](const auto& path) {
            return path.string().find(TEST_CUDNN_SHIM_V8_PLUGIN_NAME) != std::string::npos;
        });
        ASSERT_NE(it, loadedPaths.end());
        _recorder = std::make_unique<hipdnn_tests::TestPluginKnobRecorder>(*it);
    }

    void TearDown() override
    {
        _recorder.reset();
        IntegrationCudnnShimV8KnobsBase::TearDown();
    }

    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testCudnnShimV8PluginPath()};
    }

    std::unique_ptr<hipdnn_tests::TestPluginKnobRecorder> _recorder;
};

TEST_F(IntegrationCudnnShimV8Knobs, OnlyTileSizeProjected)
{
    cudnnBackendDescriptor_t engine = _builder->engine(_graph, K_KNOB_ENGINE_INDEX);
    ASSERT_EQ(knobInfoCount(engine), 1);

    std::vector<cudnnBackendDescriptor_t> infos;
    infos.reserve(CUDNN_KNOB_TYPE_COUNTS);
    for(int i = 0; i < CUDNN_KNOB_TYPE_COUNTS; ++i)
    {
        infos.push_back(_builder->create(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR));
    }
    std::int64_t filled = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(engine,
                                       CUDNN_ATTR_ENGINE_KNOB_INFO,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       CUDNN_KNOB_TYPE_COUNTS,
                                       &filled,
                                       static_cast<void*>(infos.data())),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    ASSERT_EQ(filled, 1);

    cudnnBackendKnobType_t type = CUDNN_KNOB_TYPE_COUNTS;
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(infos.front(),
                                       CUDNN_ATTR_KNOB_INFO_TYPE,
                                       CUDNN_TYPE_KNOB_TYPE,
                                       1,
                                       &count,
                                       static_cast<void*>(&type)),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(type, CUDNN_KNOB_TYPE_TILE_SIZE);
    EXPECT_EQ(GraphBuilder::getInt64(infos.front(), CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE), 0);
    EXPECT_EQ(GraphBuilder::getInt64(infos.front(), CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE), 64);
    EXPECT_EQ(GraphBuilder::getInt64(infos.front(), CUDNN_ATTR_KNOB_INFO_STRIDE), 8);
}

TEST_F(IntegrationCudnnShimV8Knobs, KnoblessEngineReportsNone)
{
    EXPECT_EQ(knobInfoCount(_builder->engine(_graph, K_CONTEXT_FAILS_INDEX)), 0);
}

TEST_F(IntegrationCudnnShimV8Knobs, TileSizeChoiceReachesPlugin)
{
    cudnnBackendDescriptor_t config
        = _builder->engineConfig(_builder->engine(_graph, K_KNOB_ENGINE_INDEX),
                                 {_builder->knobChoice(CUDNN_KNOB_TYPE_TILE_SIZE, 24)});
    ASSERT_EQ(cudnnBackendFinalize(config), CUDNN_STATUS_SUCCESS) << lastError();

    // Heuristics already queried the default config; only the plan build carries the choice.
    _recorder->reset();
    cudnnBackendDescriptor_t plan = _builder->plan(_handle, config);
    ASSERT_EQ(cudnnBackendFinalize(plan), CUDNN_STATUS_SUCCESS) << lastError();

    const auto received = _recorder->last();
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->engine_id, hipdnn_tests::plugin_constants::engineId<CudnnShimV8Plugin>());
    ASSERT_EQ(received->knobs.size(), 1U);
    EXPECT_EQ(received->knobs.front()->knob_id, "test.tile_size");
    const auto* value = received->knobs.front()->value.AsIntValue();
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->value, 24);
}

TEST_F(IntegrationCudnnShimV8Knobs, HiddenKnobChoiceRejected)
{
    EXPECT_EQ(finalizeConfigWith(K_KNOB_ENGINE_INDEX,
                                 {_builder->knobChoice(CUDNN_KNOB_TYPE_BLOCK_SIZE, 8)}),
              CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationCudnnShimV8Knobs, AbsentKnobChoiceRejected)
{
    EXPECT_EQ(finalizeConfigWith(K_KNOB_ENGINE_INDEX,
                                 {_builder->knobChoice(CUDNN_KNOB_TYPE_SPLIT_K_SLC, 1)}),
              CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationCudnnShimV8Knobs, InvalidTileSizeRejected)
{
    EXPECT_EQ(finalizeConfigWith(K_KNOB_ENGINE_INDEX,
                                 {_builder->knobChoice(CUDNN_KNOB_TYPE_TILE_SIZE, 72)}),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(finalizeConfigWith(K_KNOB_ENGINE_INDEX,
                                 {_builder->knobChoice(CUDNN_KNOB_TYPE_TILE_SIZE, 12)}),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationCudnnShimV8Knobs, DuplicateChoiceRejected)
{
    EXPECT_EQ(finalizeConfigWith(K_KNOB_ENGINE_INDEX,
                                 {_builder->knobChoice(CUDNN_KNOB_TYPE_TILE_SIZE, 8),
                                  _builder->knobChoice(CUDNN_KNOB_TYPE_TILE_SIZE, 16)}),
              CUDNN_STATUS_BAD_PARAM);
}

// Every knob of the shared knobs plugin is hidden: none maps onto a cuDNN knob type
// except test.engine_b.block_size, whose explicit valid values have no cuDNN form.
class IntegrationCudnnShimV8KnobsHidden : public IntegrationCudnnShimV8KnobsBase
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testKnobsPluginPath()};
    }

    // Index order by signed id: engine B (-9), then engine A (-8).
    static constexpr std::int64_t K_ENGINE_B_INDEX = 0;
    static constexpr std::int64_t K_ENGINE_A_INDEX = 1;
};

TEST_F(IntegrationCudnnShimV8KnobsHidden, NoKnobInfo)
{
    EXPECT_EQ(knobInfoCount(_builder->engine(_graph, K_ENGINE_A_INDEX)), 0);
    EXPECT_EQ(knobInfoCount(_builder->engine(_graph, K_ENGINE_B_INDEX)), 0);
}

TEST_F(IntegrationCudnnShimV8KnobsHidden, BlockSizeChoiceRejected)
{
    EXPECT_EQ(finalizeConfigWith(K_ENGINE_B_INDEX,
                                 {_builder->knobChoice(CUDNN_KNOB_TYPE_BLOCK_SIZE, 16)}),
              CUDNN_STATUS_NOT_SUPPORTED);
}

} // namespace

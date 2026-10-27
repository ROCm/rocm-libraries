// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The v0.x Engine and EngineConfig builders' knob projection and knob choice
// delivery against the real hipDNN backend. Engine queries need hipDNN
// heuristics, which query the device, so these tests skip on device-less hosts.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_frontend/PluginPaths.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <test_plugins/TestPluginConstants.hpp>
#include <test_plugins/TestPluginKnobRecorder.hpp>

#include "CudnnShimV8BuilderGraph.hpp"

namespace
{
namespace v8fe = hipdnn_frontend::compatibility::cudnn_frontend;
using hipdnn_shim_v8_test::appendKnob;
using hipdnn_shim_v8_test::lastError;
using hipdnn_shim_v8_test::statusOf;

class IntegrationCudnnShimV8BuilderKnobsBase : public hipdnn_shim_v8_test::ReluGraphFixture
{
protected:
    v8fe::Engine_v8 engine(std::int64_t index) const
    {
        return hipdnn_shim_v8_test::engine(*_graph, index);
    }

    static cudnnStatus_t configStatus(const v8fe::Engine_v8& built)
    {
        return statusOf([&built] {
            [[maybe_unused]] auto config = v8fe::EngineConfigBuilder_v8().setEngine(built).build();
        });
    }
};

// The shim's index orders engines by signed id: the context-failing engine (-32)
// is index 0 and the knob engine (-31) index 1.
constexpr std::int64_t K_CONTEXT_FAILS_INDEX = 0;
constexpr std::int64_t K_KNOB_ENGINE_INDEX = 1;

class IntegrationCudnnShimV8BuilderKnobs : public IntegrationCudnnShimV8BuilderKnobsBase
{
protected:
    void SetUp() override
    {
        IntegrationCudnnShimV8BuilderKnobsBase::SetUp();
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
        IntegrationCudnnShimV8BuilderKnobsBase::TearDown();
    }

    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testCudnnShimV8PluginPath()};
    }

    // The knob engine with TILE_SIZE chosen. Throws rather than writing through
    // front() of an empty list, so a lost knob fails this test instead of the binary.
    v8fe::Engine_v8 tileSizeEngine(std::int64_t tileSize) const
    {
        v8fe::Engine_v8 built = engine(K_KNOB_ENGINE_INDEX);
        if(built.getSupportedKnobs().size() != 1U)
        {
            throw std::runtime_error("knob engine reports "
                                     + std::to_string(built.getSupportedKnobs().size())
                                     + " knobs, expected 1");
        }
        built.getSupportedKnobs().front().setChoice(static_cast<std::uint64_t>(tileSize));
        return built;
    }

    std::unique_ptr<hipdnn_tests::TestPluginKnobRecorder> _recorder;
};

TEST_F(IntegrationCudnnShimV8BuilderKnobs, OnlyTileSizeProjected)
{
    v8fe::Engine_v8 built = engine(K_KNOB_ENGINE_INDEX);

    ASSERT_EQ(built.getSupportedKnobs().size(), 1U);
    const auto& knob = built.getSupportedKnobs().front();
    EXPECT_EQ(knob.getKnobType(), CUDNN_KNOB_TYPE_TILE_SIZE);
    EXPECT_EQ(knob.getMinValue(), 0);
    EXPECT_EQ(knob.getMaxValue(), 64);
    EXPECT_EQ(knob.getStride(), 8);
}

TEST_F(IntegrationCudnnShimV8BuilderKnobs, KnoblessEngineReportsNone)
{
    v8fe::Engine_v8 built = engine(K_CONTEXT_FAILS_INDEX);
    EXPECT_TRUE(built.getSupportedKnobs().empty());
}

TEST_F(IntegrationCudnnShimV8BuilderKnobs, TileSizeChoiceReachesPlugin)
{
    const v8fe::Engine_v8 built = tileSizeEngine(24);

    // Heuristics already queried the default config; only the knobbed config carries the choice.
    _recorder->reset();
    const v8fe::EngineConfig_v8 config = v8fe::EngineConfigBuilder_v8().setEngine(built).build();
    const v8fe::ExecutionPlan_v8 plan
        = v8fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
    EXPECT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << lastError();

    const auto received = _recorder->last();
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->engine_id, hipdnn_tests::plugin_constants::engineId<CudnnShimV8Plugin>());
    ASSERT_EQ(received->knobs.size(), 1U);
    EXPECT_EQ(received->knobs.front()->knob_id, "test.tile_size");
    const auto* value = received->knobs.front()->value.AsIntValue();
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->value, 24);
}

// A choice the engine did not report is sent only alongside a reported one that is set.
TEST_F(IntegrationCudnnShimV8BuilderKnobs, HiddenKnobChoiceRejected)
{
    v8fe::Engine_v8 built = tileSizeEngine(8);
    appendKnob(built, CUDNN_KNOB_TYPE_BLOCK_SIZE, 8);
    EXPECT_EQ(configStatus(built), CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationCudnnShimV8BuilderKnobs, AbsentKnobChoiceRejected)
{
    v8fe::Engine_v8 built = tileSizeEngine(8);
    appendKnob(built, CUDNN_KNOB_TYPE_SPLIT_K_SLC, 1);
    EXPECT_EQ(configStatus(built), CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationCudnnShimV8BuilderKnobs, InvalidTileSizeRejected)
{
    EXPECT_EQ(configStatus(tileSizeEngine(72)), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(configStatus(tileSizeEngine(12)), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationCudnnShimV8BuilderKnobs, DuplicateChoiceRejected)
{
    v8fe::Engine_v8 built = tileSizeEngine(8);
    appendKnob(built, CUDNN_KNOB_TYPE_TILE_SIZE, 16);
    EXPECT_EQ(configStatus(built), CUDNN_STATUS_BAD_PARAM);
}

// Every knob of the shared knobs plugin is hidden: none maps onto a cuDNN knob type
// except test.engine_b.block_size, whose explicit valid values have no cuDNN form.
class IntegrationCudnnShimV8BuilderKnobsHidden : public IntegrationCudnnShimV8BuilderKnobsBase
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

TEST_F(IntegrationCudnnShimV8BuilderKnobsHidden, NoKnobInfo)
{
    EXPECT_TRUE(engine(K_ENGINE_A_INDEX).getSupportedKnobs().empty());
    EXPECT_TRUE(engine(K_ENGINE_B_INDEX).getSupportedKnobs().empty());
}

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/SdpaGraphMutators.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#include "mocks/MockPlanBuilder.hpp"
#include "tests/engines/asm_sdpa_engine/AsmSdpaTestSupport.hpp"

#include <optional>
#include <string>
#include <vector>

namespace asm_sdpa_engine
{
namespace
{
namespace sdpa = hipdnn_test_sdk::utilities::sdpa;

using hip_kernel_provider::MockPlanBuilder;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

constexpr const char* K_ACC_KNOB = "hipkernel.sdpa.bwd.accumulator_type";

// An engine config for this engine carrying one accumulator knob setting.
flatbuffers::FlatBufferBuilder createAccumulatorKnobConfig(const std::string& value)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;
    const auto knobValue = CreateStringValueDirect(builder, value.c_str());
    const std::vector<flatbuffers::Offset<KnobSetting>> knobSettings{
        CreateKnobSettingDirect(builder, K_ACC_KNOB, KnobValue::StringValue, knobValue.Union())};
    builder.Finish(CreateEngineConfigDirect(builder, AsmSdpaEngine::staticId(), &knobSettings));
    return builder;
}

// The accumulator knob set to an integer instead of a string.
flatbuffers::FlatBufferBuilder createIntAccumulatorKnobConfig()
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;
    const auto knobValue = CreateIntValue(builder, 16);
    const std::vector<flatbuffers::Offset<KnobSetting>> knobSettings{
        CreateKnobSettingDirect(builder, K_ACC_KNOB, KnobValue::IntValue, knobValue.Union())};
    builder.Finish(CreateEngineConfigDirect(builder, AsmSdpaEngine::staticId(), &knobSettings));
    return builder;
}

flatbuffers::FlatBufferBuilder createEngineConfigWithoutKnobs()
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(
        builder, AsmSdpaEngine::staticId()));
    return builder;
}

EngineConfigWrapper wrapConfig(const flatbuffers::FlatBufferBuilder& builder)
{
    return EngineConfigWrapper(builder.GetBufferPointer(), builder.GetSize());
}

// The packed [4, 8, 256, 128] bf16 graphs both builders serve.
flatbuffers::FlatBufferBuilder createSdpaBwdGraph()
{
    return createPackedSdpaBwdGraph({4, 8, 256, 128}, {4, 8, 256, 128});
}

flatbuffers::FlatBufferBuilder createSdpaFwdGraph()
{
    return createPackedSdpaFwdGraph({4, 8, 256, 128}, {4, 8, 256, 128});
}

hipdnn_flatbuffers_sdk::data_objects::KnobT makeStringKnob(const std::string& id)
{
    hipdnn_flatbuffers_sdk::data_objects::KnobT knob;
    knob.knob_id = id;
    hipdnn_flatbuffers_sdk::data_objects::StringValueT defaultValue;
    defaultValue.value = "default";
    knob.default_value.Set(defaultValue);
    return knob;
}

class TestAsmSdpaEngine : public ::testing::Test
{
protected:
    AsmSdpaEngine _engine;
    Handle _handle;

    // Registered in the same order as the provider's engine container.
    void SetUp() override
    {
        _engine.addPlanBuilder(std::make_unique<SdpaFwdPlanBuilder>());
        _engine.addPlanBuilder(std::make_unique<SdpaBwdPlanBuilder>());
    }

    // True when the device runs the backward engine; call after SKIP_IF_NO_DEVICES().
    bool deviceRunsBwd() const
    {
        return bwd_dispatch::isSupportedArch(
            hip_kernel_provider_common::getDeviceString(_handle.getStream()));
    }

    EngineDetailsWrapper details(const GraphWrapper& graph)
    {
        hipdnnPluginConstData_t detailsOut;
        _engine.getDetails(_handle, graph, detailsOut);
        return EngineDetailsWrapper(detailsOut.ptr, detailsOut.size);
    }
};

TEST_F(TestAsmSdpaEngine, GetDetailsReportsGraphCaptureSupport)
{
    const hipdnn_test_sdk::utilities::MockGraph mockGraph;
    hipdnnPluginConstData_t result;
    _engine.getDetails(_handle, mockGraph, result);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper details(result.ptr,
                                                                                     result.size);
    EXPECT_THAT(
        details.behaviorNotes(),
        ::testing::Contains(static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE)));
}

TEST_F(TestAsmSdpaEngine, IsApplicableReturnsFalseForNonSdpaGraph)
{
    // Create a batchnorm inference graph - this does not use SDPA attributes
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_FALSE(_engine.isApplicable(_handle, graphWrapper));
}

TEST_F(TestAsmSdpaEngine, IsApplicableReturnsTrueForSdpaGraph)
{
    SKIP_IF_NO_DEVICES();

    const auto deviceString = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaFwdGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_engine.isApplicable(_handle, graphWrapper));
}

TEST_F(TestAsmSdpaEngine, IsApplicableReturnsTrueForRaggedSdpaGraph)
{
    SKIP_IF_NO_DEVICES();

    const auto deviceString = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(deviceString != "gfx942" && deviceString != "gfx950")
    {
        GTEST_SKIP();
    }

    const std::vector<int64_t> dims{4, 8, 256, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, {3, 1, 2, 0});
    auto builder
        = sdpa::withRaggedTensors(hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
                                      dims,
                                      strides,
                                      dims,
                                      strides,
                                      dims,
                                      strides,
                                      dims,
                                      strides,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                      hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT),
                                  {"q", "k", "v", "o"});

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_engine.isApplicable(_handle, graphWrapper));
}

TEST_F(TestAsmSdpaEngine, GetDetailsReportsKnobsOfApplicableBuilderOnly)
{
    auto declining = std::make_unique<::testing::NiceMock<MockPlanBuilder>>();
    ON_CALL(*declining, isApplicable).WillByDefault(::testing::Return(false));
    ON_CALL(*declining, getCustomKnobs)
        .WillByDefault(::testing::Return(
            std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>{makeStringKnob("declining")}));

    auto serving = std::make_unique<::testing::NiceMock<MockPlanBuilder>>();
    ON_CALL(*serving, isApplicable).WillByDefault(::testing::Return(true));
    ON_CALL(*serving, getCustomKnobs)
        .WillByDefault(::testing::Return(
            std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>{makeStringKnob("serving")}));

    AsmSdpaEngine engine;
    engine.addPlanBuilder(std::move(declining));
    engine.addPlanBuilder(std::move(serving));

    const hipdnn_test_sdk::utilities::MockGraph graph;
    hipdnnPluginConstData_t detailsOut;
    engine.getDetails(_handle, graph, detailsOut);
    const EngineDetailsWrapper engineDetails(detailsOut.ptr, detailsOut.size);

    ASSERT_EQ(engineDetails.knobCount(), 1u);
    EXPECT_EQ(engineDetails.getKnobByName("serving").knobId(), "serving");
}

TEST_F(TestAsmSdpaEngine, InitializeExecutionContextBuildsPlanWithConfiguredSettings)
{
    auto builder = std::make_unique<::testing::NiceMock<MockPlanBuilder>>();
    ON_CALL(*builder, isApplicable).WillByDefault(::testing::Return(true));
    ON_CALL(*builder, initializeExecutionSettings)
        .WillByDefault([](const Handle&,
                          const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&,
                          const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&,
                          Settings& settings) { settings.accumulatorType = AccumulatorType::A16; });
    std::optional<AccumulatorType> accumulatorSeenByBuildPlan;
    ON_CALL(*builder, buildPlan)
        .WillByDefault([&](const Handle&,
                           const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph&,
                           const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&,
                           Context& context) {
            accumulatorSeenByBuildPlan = context.executionSettings().accumulatorType;
        });

    AsmSdpaEngine engine;
    engine.addPlanBuilder(std::move(builder));

    const hipdnn_test_sdk::utilities::MockGraph graph;
    const auto configBuilder = createEngineConfigWithoutKnobs();
    Context context;
    engine.initializeExecutionContext(_handle, graph, wrapConfig(configBuilder), context);

    EXPECT_EQ(accumulatorSeenByBuildPlan, AccumulatorType::A16);
}

TEST_F(TestAsmSdpaEngine, GetDetailsAdvertisesAccumulatorKnobForBwdGraph)
{
    SKIP_IF_NO_DEVICES();
    if(!deviceRunsBwd())
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto engineDetails = details(graph);

    ASSERT_EQ(engineDetails.knobCount(), 1u);
    EXPECT_EQ(engineDetails.getKnobByName(K_ACC_KNOB).knobId(), K_ACC_KNOB);
}

TEST_F(TestAsmSdpaEngine, GetDetailsAdvertisesNoKnobsForFwdGraph)
{
    SKIP_IF_NO_DEVICES();
    if(!deviceRunsBwd())
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaFwdGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    ASSERT_TRUE(_engine.isApplicable(_handle, graph));

    EXPECT_EQ(details(graph).knobCount(), 0u);
}

TEST_F(TestAsmSdpaEngine, MaxWorkspaceSizeFollowsAccumulatorKnob)
{
    SKIP_IF_NO_DEVICES();
    if(!deviceRunsBwd())
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
    const auto noKnobs = createEngineConfigWithoutKnobs();
    const auto a32 = createAccumulatorKnobConfig("a32");
    const auto a16 = createAccumulatorKnobConfig("a16");

    const size_t defaultSize = _engine.getMaxWorkspaceSize(_handle, graph, wrapConfig(noKnobs));
    const size_t a32Size = _engine.getMaxWorkspaceSize(_handle, graph, wrapConfig(a32));
    const size_t a16Size = _engine.getMaxWorkspaceSize(_handle, graph, wrapConfig(a16));

    // [4, 8, 256, 128] bf16: D buffer 4*8*256*4 = 32768; A32 dq_acc
    // 4*8*256*128*4 = 4194304; the A16 dq_acc (gfx950 only, read by its
    // dq_shuffle post stage) 4*8*256*128*2 = 2097152.
    EXPECT_EQ(a32Size, 32768u + 4194304u);
    EXPECT_EQ(defaultSize, a32Size);
    const bool a16HasDqAcc
        = hip_kernel_provider_common::getDeviceString(_handle.getStream()) == "gfx950";
    EXPECT_EQ(a16Size, a16HasDqAcc ? 32768u + 2097152u : 32768u);
}

TEST_F(TestAsmSdpaEngine, InvalidAccumulatorKnobIsRejected)
{
    SKIP_IF_NO_DEVICES();
    if(!deviceRunsBwd())
    {
        GTEST_SKIP();
    }

    auto builder = createSdpaBwdGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    for(const auto& configBuilder :
        {createAccumulatorKnobConfig("a64"), createIntAccumulatorKnobConfig()})
    {
        const auto config = wrapConfig(configBuilder);
        EXPECT_THROW(_engine.getMaxWorkspaceSize(_handle, graph, config),
                     hipdnn_plugin_sdk::HipdnnPluginException);
        Context context;
        EXPECT_THROW(_engine.initializeExecutionContext(_handle, graph, config, context),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
}

} // namespace
} // namespace asm_sdpa_engine

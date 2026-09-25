// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"

namespace asm_sdpa_engine
{
namespace
{

using hipdnn_flatbuffers_sdk::data_objects::PredictionKind;
using hipdnn_flatbuffers_sdk::data_objects::PredictionStatus;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

/// An engine configuration for this engine that names @p metric (RFC 0019 §11.4).
flatbuffers::FlatBufferBuilder engineConfigNaming(const char* metric)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfigDirect(
        builder, AsmSdpaEngine::staticId(), nullptr, metric));
    return builder;
}

flatbuffers::FlatBufferBuilder sdpaFwdGraph()
{
    const std::vector<int64_t> dims{4, 8, 256, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    return hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        dims,
        strides,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
}

class TestAsmSdpaEngine : public ::testing::Test
{
protected:
    AsmSdpaEngine _engine;
    Handle _handle;

    void SetUp() override
    {
        _engine.addPlanBuilder(std::make_unique<SdpaFwdPlanBuilder>());
    }
};

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

    auto builder = sdpaFwdGraph();
    const GraphWrapper graphWrapper(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_engine.isApplicable(_handle, graphWrapper));
}

/// Every answer carries the metric it was asked in, the decline included: the backend
/// rejects a response whose metric differs from the request, so a decline without one
/// would read as a broken engine rather than an absent model.
TEST_F(TestAsmSdpaEngine, ConfigurationDeclineCarriesTheRequestedMetric)
{
    auto graphBuilder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    auto configBuilder = engineConfigNaming("time");
    const EngineConfigWrapper timeConfig(configBuilder.GetBufferPointer(), configBuilder.GetSize());
    const auto inTime = _engine.getPrediction(
        _handle, graph, timeConfig, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);
    EXPECT_EQ(inTime.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(inTime.kind, PredictionKind::CONFIGURATION);
    EXPECT_EQ(inTime.metric, "time");

    // A configuration that names nothing asks in the default metric.
    const EngineConfigWrapper unnamed(nullptr, 0);
    EXPECT_EQ(
        _engine.getPrediction(_handle, graph, unnamed, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true)
            .metric,
        "tflops");
}

/// RFC 0019 §4.4: no metric substitution. This engine ships throughput models only, so a
/// request in `time` is unanswered even on an architecture whose `tflops` model is
/// deployed -- converting one into the other would invent a number no model predicted.
TEST_F(TestAsmSdpaEngine, NoModelForTheRequestedMetricIsUnavailableNotSubstituted)
{
    SKIP_IF_NO_DEVICES();

    auto graphBuilder = sdpaFwdGraph();
    const GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    auto configBuilder = engineConfigNaming("time");
    const EngineConfigWrapper config(configBuilder.GetBufferPointer(), configBuilder.GetSize());

    const auto prediction
        = _engine.getPrediction(_handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true);
    EXPECT_EQ(prediction.status, PredictionStatus::UNAVAILABLE) << prediction.reason;
    EXPECT_EQ(prediction.metric, "time");
}

/// A metric the registry does not know has no direction to rank by. It is a bad request,
/// not a missing model, so it must not come back looking like an ordinary UNAVAILABLE.
TEST_F(TestAsmSdpaEngine, UnregisteredMetricIsABadRequest)
{
    auto graphBuilder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    auto configBuilder = engineConfigNaming("bandwidth");
    const EngineConfigWrapper config(configBuilder.GetBufferPointer(), configBuilder.GetSize());

    EXPECT_THROW(static_cast<void>(_engine.getPrediction(
                     _handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

} // namespace
} // namespace asm_sdpa_engine

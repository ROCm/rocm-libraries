// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#include "mocks/MockPlanBuilder.hpp"
#include "mocks/StampedGraph.hpp"

namespace asm_sdpa_engine
{
namespace
{

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

    const std::vector<int64_t> dims{4, 8, 256, 128};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    auto builder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph(
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

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    EXPECT_TRUE(_engine.isApplicable(_handle, graphWrapper));
}

// The plugin advertises a newer engine API than this engine is written against, so the
// engine itself declines ragged, custom-alignment and token-unit ragged graphs before any
// plan builder sees them.
TEST(TestAsmSdpaEngineApiFloor, DeclinesGraphsNeedingANewerEngineApi)
{
    auto planBuilder
        = std::make_unique<::testing::NiceMock<hip_kernel_provider::MockPlanBuilder>>();
    ON_CALL(*planBuilder, isApplicable(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(true));
    AsmSdpaEngine engine;
    engine.addPlanBuilder(std::move(planBuilder));

    Handle handle;
    using hip_kernel_provider::StampedGraph;
    EXPECT_TRUE(engine.isApplicable(
        handle, StampedGraph(hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION)));
    EXPECT_FALSE(engine.isApplicable(
        handle, StampedGraph(hipdnn_plugin_sdk::K_RAGGED_TENSOR_MIN_API_VERSION)));
    EXPECT_FALSE(engine.isApplicable(
        handle, StampedGraph(hipdnn_plugin_sdk::K_RAGGED_OFFSET_MULTIPLIER_MIN_VERSION)));
}

} // namespace
} // namespace asm_sdpa_engine

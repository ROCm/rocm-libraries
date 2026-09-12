// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <set>
#include <string>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"

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

/// RFC 0019 Open Question 7 (RESOLVED) plus §11.2: with nothing deployed for the UUIDs
/// this engine declares -- the state of every machine that has not installed a model --
/// the engine answers UNAVAILABLE. Absence is a normal outcome, never an exception and
/// never a crash, whether or not a descriptor tree exists to look in.
TEST_F(TestAsmSdpaEngine, ReportsNoEstimateWhenNoDeclaredModelIsDeployed)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(nullptr, 0);

    hipdnn_flatbuffers_sdk::data_objects::EnginePredictionT prediction;
    ASSERT_NO_THROW(prediction = _engine.getPrediction(
                        _handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true));
    EXPECT_EQ(prediction.status,
              hipdnn_flatbuffers_sdk::data_objects::PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(prediction.engine_id, AsmSdpaEngine::staticId());
    EXPECT_EQ(prediction.kind, hipdnn_flatbuffers_sdk::data_objects::PredictionKind::ENGINE);
}

/// ASM SDPA runs its own kernel selection, so it has no exact configuration of ours to
/// predict: RFC 0019 §11.2's "A only (opaque)" row. Declining must stay a decline rather
/// than becoming an error now that the engine answers the ENGINE query.
TEST_F(TestAsmSdpaEngine, DeclinesTheConfigurationPredictionQuery)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(nullptr, 0);

    const auto prediction = _engine.getPrediction(
        _handle, graph, config, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);
    EXPECT_EQ(prediction.status,
              hipdnn_flatbuffers_sdk::data_objects::PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(prediction.kind, hipdnn_flatbuffers_sdk::data_objects::PredictionKind::CONFIGURATION);
}

/// The declaration surface itself. A malformed or duplicated literal would not fail the
/// build -- it would silently mean "this architecture never binds a model", or "gfx950
/// answers with gfx942's model" -- so the compiled-in table is checked here.
TEST(TestAsmSdpaEngineDeclaration, DeclaredModelIdsAreDistinctWellFormedUuids)
{
    std::set<std::string> seenArch;
    std::set<std::string> seenId;
    EXPECT_FALSE(AsmSdpaEngine::L1_MODEL_IDS.empty());
    for(const auto& [arch, id] : AsmSdpaEngine::L1_MODEL_IDS)
    {
        EXPECT_FALSE(arch.empty());
        EXPECT_NO_THROW(static_cast<void>(hipdnn_flatbuffers_sdk::utilities::parseUuid(id))) << id;
        EXPECT_TRUE(seenArch.insert(std::string(arch)).second) << arch;
        EXPECT_TRUE(seenId.insert(std::string(id)).second) << "two architectures declare " << id;
    }
}

} // namespace
} // namespace asm_sdpa_engine

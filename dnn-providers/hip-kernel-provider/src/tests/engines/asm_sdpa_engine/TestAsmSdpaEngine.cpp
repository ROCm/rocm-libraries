// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <string_view>

#include "AsmSdpaSelectorRevisionFixtures.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#include "version.h"

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

/// RFC 0019 §11.2: with no declared model deployed, the engine answers UNAVAILABLE rather
/// than throwing, whether or not a descriptor tree exists.
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

/// A model whose recorded revision mismatches is refused, so the revision must track the
/// forward kernels and dispatch sources, not the provider release.
TEST(TestAsmSdpaEngineDeclaration, TheSelectorRevisionNamesTheForwardSurfaceAndNotTheRelease)
{
    const std::string revision = AsmSdpaEngine::selectorRevision();
    const std::string prefix = "hip-kernel-provider/asm-sdpa-fwd/";
    ASSERT_EQ(revision.rfind(prefix, 0), 0u) << revision;

    // "undetermined" here means the CMake wiring that computes the digest was dropped.
    const std::string digest = revision.substr(prefix.size());
    EXPECT_EQ(digest.size(), 16u) << revision;
    EXPECT_TRUE(std::all_of(digest.begin(), digest.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
    })) << revision;

    // No provider version component, in any form.
    EXPECT_EQ(revision.find(HIP_KERNEL_PROVIDER_VERSION_STRING), std::string::npos) << revision;
    EXPECT_EQ(revision.find("0.2."), std::string::npos) << revision;
}

/// The backend rejects a response whose metric differs from the request, so even a decline
/// must carry the requested metric.
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

/// RFC 0019 §4.4: no metric substitution. Only `tflops` models ship, so `time` stays
/// unanswered even where a `tflops` model is deployed.
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

/// An unregistered metric is a bad request, not a missing model, so it throws rather than
/// returning UNAVAILABLE.
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

namespace fixtures = selector_revision_fixtures;

constexpr std::string_view SELECTOR_REVISION_PREFIX = "hip-kernel-provider/asm-sdpa-fwd/";

/// The digest part of the revision this build reports.
std::string_view builtDigest()
{
    std::string_view revision = AsmSdpaEngine::selectorRevision();
    if(revision.substr(0, SELECTOR_REVISION_PREFIX.size()) != SELECTOR_REVISION_PREFIX)
    {
        return {};
    }
    revision.remove_prefix(SELECTOR_REVISION_PREFIX.size());
    return revision;
}

/// Shipped models record the revision verbatim, so its form is a contract.
TEST(TestAsmSdpaSelectorRevision, IsTheProviderPrefixAndSixteenLowercaseHexDigits)
{
    const std::string_view digest = builtDigest();
    ASSERT_EQ(digest.size(), 16u) << AsmSdpaEngine::selectorRevision();
    for(const char c : digest)
    {
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))
            << AsmSdpaEngine::selectorRevision();
    }
}

/// The fixtures are this engine's tree written with LF and with CRLF; both must report the
/// build's own revision, or Windows and Linux builds refuse each other's models.
TEST(TestAsmSdpaSelectorRevision, CrlfAndLfCheckoutsReportTheBuiltRevision)
{
    EXPECT_EQ(std::string_view(fixtures::REVISION_CRLF), std::string_view(fixtures::REVISION_LF));
    EXPECT_EQ(builtDigest(), std::string_view(fixtures::REVISION_LF));
}

/// Each forward selection input must move the revision; everything else must not.
TEST(TestAsmSdpaSelectorRevision, MovesForEveryForwardSelectionInputAndNothingElse)
{
    const std::string_view base(fixtures::REVISION_LF);
    for(const auto& probe : fixtures::PROBES)
    {
        EXPECT_EQ(std::string_view(probe.revision) != base, probe.mustExpire)
            << probe.change << (probe.mustExpire ? " must" : " must not")
            << " change the forward selector revision";
    }
}

} // namespace
} // namespace asm_sdpa_engine

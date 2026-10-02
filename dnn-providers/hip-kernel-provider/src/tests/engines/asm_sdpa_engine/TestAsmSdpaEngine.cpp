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

/// The expiry rule every shipped L1 model is judged against.
///
/// A model records this string and the loader refuses one that does not match, so what the
/// revision NAMES decides which changes expire a model. It used to name the provider
/// release, which is wrong in both directions: `0.2.0 -> 0.2.1` for a change that cannot
/// touch this engine expired both shipped models, and a vendored kernel swap under a fixed
/// version expired nothing -- the silent direction, since a stale L1 estimate changes which
/// ENGINE is selected. It is now a digest over the forward kernels, the CSVs that describe
/// them and the forward dispatch sources.
TEST(TestAsmSdpaEngineDeclaration, TheSelectorRevisionNamesTheForwardSurfaceAndNotTheRelease)
{
    const std::string revision = AsmSdpaEngine::selectorRevision();
    const std::string prefix = "hip-kernel-provider/asm-sdpa-fwd/";
    ASSERT_EQ(revision.rfind(prefix, 0), 0u) << revision;

    // A build that did not compute the digest reports "undetermined", which no shipped
    // model can match -- deliberately, but it means the CMake wiring has been dropped.
    const std::string digest = revision.substr(prefix.size());
    EXPECT_EQ(digest.size(), 16u) << revision;
    EXPECT_TRUE(std::all_of(digest.begin(), digest.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
    })) << revision;

    // The regression itself: no provider version component, in any form.
    EXPECT_EQ(revision.find(HIP_KERNEL_PROVIDER_VERSION_STRING), std::string::npos) << revision;
    EXPECT_EQ(revision.find("0.2."), std::string::npos) << revision;
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

/// A shipped model records the revision verbatim, so its form is a contract: a build that
/// did not compute the digest reports "undetermined", which no model can match.
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

/// One commit is one selector, whatever the checkout's line endings: a Windows (CRLF) and a
/// Linux (LF) checkout that disagree each refuse the models the other trained. The
/// fixtures are this engine's tree written both ways, and the build's own revision is the
/// one both must report.
TEST(TestAsmSdpaSelectorRevision, CrlfAndLfCheckoutsReportTheBuiltRevision)
{
    EXPECT_EQ(std::string_view(fixtures::REVISION_CRLF), std::string_view(fixtures::REVISION_LF));
    EXPECT_EQ(builtDigest(), std::string_view(fixtures::REVISION_LF));
}

/// The revision expires every deployed model, so it must move for each change to what
/// decides the forward kernel or its arguments (else a stale L1 estimate picks the engine),
/// and for nothing else (else every model expires for a change that cannot touch it).
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

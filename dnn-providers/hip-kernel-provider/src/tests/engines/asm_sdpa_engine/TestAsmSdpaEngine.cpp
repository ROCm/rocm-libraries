// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/MockGraph.hpp>
#include <hipdnn_test_sdk/utilities/SdpaGraphMutators.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "AsmSdpaSelectorRevisionFixtures.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "core/Settings.hpp"
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#include "mocks/MockPlanBuilder.hpp"
#include "tests/engines/ShippedL1ModelGate.hpp"
#include "tests/engines/asm_sdpa_engine/AsmSdpaTestSupport.hpp"
#include "version.h"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "AsmSdpaShippedModels.hpp"
#endif

namespace asm_sdpa_engine
{
namespace
{
namespace sdpa = hipdnn_test_sdk::utilities::sdpa;

using hip_kernel_provider::MockPlanBuilder;
using hipdnn_flatbuffers_sdk::data_objects::PredictionKind;
using hipdnn_flatbuffers_sdk::data_objects::PredictionStatus;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

constexpr const char* K_ACC_KNOB = "hipkernel.sdpa.bwd.accumulator_type";

/// An engine configuration for this engine that names @p metric (RFC 0019 §11.4).
flatbuffers::FlatBufferBuilder engineConfigNaming(const char* metric)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfigDirect(
        builder, AsmSdpaEngine::staticId(), nullptr, metric));
    return builder;
}

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
    // getDetails asks the builders whether they serve the graph, so it needs a real
    // graph; the behavior note does not depend on which builder serves it.
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THAT(
        details(graph).behaviorNotes(),
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
/// forward kernels and selection sources, not the provider release.
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

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
/// A shipped model that records another revision is refused at load and the engine reports
/// no estimate, silently; this is where that fails the build.
TEST(TestAsmSdpaEngineDeclaration, ShippedL1ModelsRecordTheRevisionThisBuildComputes)
{
    EXPECT_TRUE(hip_kernel_provider::shipped_model_gate::shippedL1ModelsBindThisBuild(
        AsmSdpaEngine::engineName(),
        "asm_sdpa_fwd",
        shipped_models::MODELS,
        HKP_SELECTOR_DUMP_DIR,
        AsmSdpaEngine::L1_MODEL_IDS,
        AsmSdpaEngine::selectorRevision()));
}
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

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

    auto graphBuilder = createSdpaFwdGraph();
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

TEST_F(TestAsmSdpaEngine, InitializeExecutionContextThrowsInternalErrorIfNoBuilderApplies)
{
    // A graph no builder accepts must not yield a context that has no plan.
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    Context context;
    const hipdnn_test_sdk::utilities::MockEngineConfig engineConfig;

    try
    {
        _engine.initializeExecutionContext(_handle, graphWrapper, engineConfig, context);
        FAIL() << "a context with no plan must not be created";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    }
}

TEST_F(TestAsmSdpaEngine, GetMaxWorkspaceSizeThrowsInternalErrorIfNoBuilderApplies)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());
    const hipdnn_test_sdk::utilities::MockEngineConfig engineConfig;

    try
    {
        static_cast<void>(_engine.getMaxWorkspaceSize(_handle, graphWrapper, engineConfig));
        FAIL() << "a workspace query with no applicable builder must throw";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
    }
}

} // namespace
} // namespace asm_sdpa_engine

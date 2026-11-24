// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
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
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
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

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "AsmSdpaShippedModels.hpp"
#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>
#include <nlohmann/json.hpp>
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

// An engine config for this engine carrying one integer knob setting.
flatbuffers::FlatBufferBuilder createIntKnobConfig(const char* knobName, int64_t value)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;
    const auto knobValue = CreateIntValue(builder, value);
    const std::vector<flatbuffers::Offset<KnobSetting>> knobSettings{
        CreateKnobSettingDirect(builder, knobName, KnobValue::IntValue, knobValue.Union())};
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

// The status a call throws, or SUCCESS when it returns.
template <typename Call>
hipdnnPluginStatus_t statusOf(Call&& call)
{
    try
    {
        call();
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        return error.getStatus();
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
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

// A plan builder that serves every graph.
std::unique_ptr<IPlanBuilder> servingPlanBuilder()
{
    auto builder = std::make_unique<::testing::NiceMock<MockPlanBuilder>>();
    ON_CALL(*builder, isApplicable).WillByDefault(::testing::Return(true));
    return builder;
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

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
/// A shipped model that records another revision is refused at load and the engine reports
/// no estimate, silently; one declared for an architecture other than the one it ships for
/// is bound there and its estimate refused, leaving that architecture unranked; and a
/// declared architecture missing from the staged model list ships no model. This is where
/// any of them fails the build.
TEST(TestAsmSdpaEngineDeclaration, ShippedL1ModelsRecordTheRevisionThisBuildComputes)
{
    EXPECT_TRUE(hip_kernel_provider::shipped_model_gate::shippedL1ModelsBindThisBuild(
        AsmSdpaEngine::engineName(),
        "dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/"
        "AsmSdpaSelectorRevision.cmake",
        shipped_models::MODELS,
        shipped_models::FILES,
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

/// A configuration request for a graph a builder serves is answered by the CONFIGURATION
/// branch, not by a model: this engine selects its own kernel and has no exact configuration
/// to name. The builder serves the graph, so the answer cannot be the "no plan" decline.
TEST_F(TestAsmSdpaEngine, ConfigurationRequestForAServedGraphDeclinesWithTheExactReason)
{
    AsmSdpaEngine engine;
    engine.addPlanBuilder(servingPlanBuilder());
    const hipdnn_test_sdk::utilities::MockGraph graph;
    const EngineConfigWrapper config(nullptr, 0);

    const auto prediction = engine.getPrediction(
        _handle, graph, config, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);
    EXPECT_EQ(prediction.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(prediction.kind, PredictionKind::CONFIGURATION);
    EXPECT_EQ(prediction.metric, "tflops");
    EXPECT_EQ(prediction.reason,
              "ASM SDPA selects its own kernel and predicts no exact configuration");
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

/// Shipped models record the revision verbatim, so its form is a contract: the recipe's
/// prefix and sixteen lowercase hex digits. A revision of "undetermined" means the wiring
/// that hands the recipe's revision to the engine was dropped.
TEST(TestAsmSdpaSelectorRevision, IsTheProviderPrefixAndSixteenLowercaseHexDigits)
{
    std::string_view digest = AsmSdpaEngine::selectorRevision();
    const std::string_view prefix(fixtures::REVISION_PREFIX);
    ASSERT_EQ(digest.substr(0, prefix.size()), prefix) << AsmSdpaEngine::selectorRevision();
    digest.remove_prefix(prefix.size());
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
    EXPECT_EQ(std::string_view(AsmSdpaEngine::selectorRevision()),
              std::string_view(fixtures::REVISION_LF));
}

/// Each selection input, of either direction, must move the revision, in content and in the
/// path it is keyed by (a kernel moved to another arch directory); everything else must not.
TEST(TestAsmSdpaSelectorRevision, MovesForEverySelectionInputAndNothingElse)
{
    const std::string_view base(fixtures::REVISION_LF);
    for(const auto& probe : fixtures::PROBES)
    {
        EXPECT_EQ(std::string_view(probe.revision) != base, probe.mustExpire)
            << probe.change << (probe.mustExpire ? " must" : " must not")
            << " change the selector revision";
    }
}

/// A file added to the engine tree is neither hashed nor probed until someone decides it
/// is an input; the probes above only cover the files the recipe lists.
TEST(TestAsmSdpaSelectorRevision, EveryEngineFileIsAnInputOrDeclaredNotToBe)
{
    EXPECT_EQ(std::string_view(fixtures::UNCLASSIFIED_FILES), "")
        << "Add each to hkp_asm_sdpa_sources() (it decides a kernel or its launch) or to "
           "hkp_asm_sdpa_not_inputs() (it does not) in AsmSdpaSelectorRevision.cmake";
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

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
/// A device as the shipped models' training rows recorded it: an MI300X for gfx942, an
/// MI355X for gfx950, in the properties UHD device features read.
hipDeviceProp_t trainingDevice(const std::string& arch)
{
    hipDeviceProp_t device{};
    std::snprintf(device.gcnArchName, sizeof(device.gcnArchName), "%s", arch.c_str());
    device.warpSize = 64;
    device.memoryBusWidth = 8192;
    const bool mi300x = arch == "gfx942";
    device.multiProcessorCount = mi300x ? 304 : 256;
    device.memoryClockRate = mi300x ? 1300000 : 2000000;
    device.sharedMemPerBlock = mi300x ? 65536 : 163840;
    device.totalGlobalMem = mi300x ? 206141652992 : 309220868096;
    return device;
}

/// An engine that estimates for a synthesized device instead of the handle stream's, to see
/// what each architecture's model answers on a host of any architecture.
class SynthesizedDeviceEngine : public AsmSdpaEngine
{
public:
    explicit SynthesizedDeviceEngine(const hipDeviceProp_t& device)
        : _device(device)
    {
    }

    SynthesizedDeviceEngine(const hipDeviceProp_t& device,
                            const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& catalog)
        : AsmSdpaEngine(catalog)
        , _device(device)
    {
    }

protected:
    const hipDeviceProp_t& predictionDevice(Handle& /*handle*/) const override
    {
        return _device;
    }

private:
    hipDeviceProp_t _device;
};

/// The models describe the graphs the builders serve in the three dense layouts the corpus
/// covers; a graph no builder serves gets no estimate instead of an extrapolated one, before
/// any device or model is consulted. A stride pattern a builder serves outside those layouts
/// is estimated by extrapolation, which this test does not cover.
TEST_F(TestAsmSdpaEngine, GetPredictionDeclinesAGraphNoBuilderServes)
{
    auto declining = std::make_unique<::testing::NiceMock<MockPlanBuilder>>();
    ON_CALL(*declining, isApplicable).WillByDefault(::testing::Return(false));
    AsmSdpaEngine engine;
    engine.addPlanBuilder(std::move(declining));

    const hipdnn_test_sdk::utilities::MockGraph graph;
    const EngineConfigWrapper config(nullptr, 0);
    const auto prediction
        = engine.getPrediction(_handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true);
    EXPECT_EQ(prediction.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(prediction.reason, "ASM SDPA serves no plan for this graph");
}

/// A negative `global.workspace_size_limit` is a malformed request on the prediction path
/// too: it throws INVALID_VALUE, as getMaxWorkspaceSize and initializeExecutionContext do,
/// instead of reading as a missing estimate.
TEST_F(TestAsmSdpaEngine, ANegativeWorkspaceLimitIsAnInvalidValueOnThePredictionPath)
{
    SynthesizedDeviceEngine engine(trainingDevice("gfx942"));
    engine.addPlanBuilder(servingPlanBuilder());
    auto graphBuilder = createSdpaFwdGraph();
    const GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const auto negative
        = createIntKnobConfig(hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, -1);
    EXPECT_EQ(statusOf([&] {
                  static_cast<void>(engine.getPrediction(
                      _handle, graph, wrapConfig(negative), HIPDNN_ENGINE_PREDICTION_ENGINE, true));
              }),
              HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
}

/// The gfx950 model reads the device's memory clock and bandwidth, so a device that reports
/// no clock gets no estimate instead of a number the model produces from a value it never
/// trained on.
TEST_F(TestAsmSdpaEngine, ADeviceThatReportsNoMemoryClockHasNoEstimate)
{
    auto device = trainingDevice("gfx950");
    device.memoryClockRate = 0;
    SynthesizedDeviceEngine engine(device);
    engine.addPlanBuilder(servingPlanBuilder());
    auto graphBuilder = createSdpaFwdGraph();
    const GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const EngineConfigWrapper config(nullptr, 0);

    const auto prediction
        = engine.getPrediction(_handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true);
    EXPECT_EQ(prediction.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(prediction.metric, "tflops");
    EXPECT_THAT(prediction.reason, ::testing::HasSubstr("memory clock"));
}

#ifdef HIPKERNELPROVIDER_ASM_SDPA_DESCRIPTOR_RELDIR
/// Dense bf16 peak in TFLOPS, AMD's published MI300X and MI355X figures. No estimate of an
/// attention graph can exceed it.
double denseBf16PeakTflops(const std::string& arch)
{
    return arch == "gfx942" ? 1307.4 : 2516.6;
}

/// The architectures this install carries production content for: the shard directories
/// (arch_content/hip-kernel-provider/<arch>/) beside asm_sdpa/. TheRock splits the provider's
/// per-arch content into one artifact per GPU family, so a gfx950 job holds gfx950's model
/// and not gfx942's, and a family with no production content (gfx103X, gfx117X) holds no
/// shard and neither model.
std::set<std::string> installedShardArches(const std::filesystem::path& productRoot)
{
    std::set<std::string> arches;
    std::error_code error;
    for(const auto& entry : std::filesystem::directory_iterator(productRoot, error))
    {
        const auto name = entry.path().filename().string();
        if(name.rfind("gfx", 0) == 0 && entry.is_directory(error))
        {
            arches.insert(name);
        }
    }
    return arches;
}

/// Whether this binary runs from the build tree that staged the models, where every declared
/// model is there whatever the packaging state: the provider copies each one itself when
/// production packaging is dormant, and the product pack stages each one when it is active.
/// Only the binary's own directory is compared with the one it was built into; nothing is
/// read from the build tree, so an installed or relocated copy answers false.
bool runsFromItsBuildTree()
{
    std::error_code error;
    return std::filesystem::equivalent(hip_kernel_provider::testing::descriptorSetRoot("."),
                                       HIPKERNELPROVIDER_ASM_SDPA_TEST_BUILD_DIR,
                                       error);
}

/// Whether the model declared for @p arch may be missing from this install without that being
/// a failure: outside the build tree, only where the install carries no shard for the
/// architecture and no model folder either.
bool modelMayBeAbsent(const std::string& arch)
{
    const auto modelsRoot = hip_kernel_provider::testing::descriptorSetRoot(
        HIPKERNELPROVIDER_ASM_SDPA_DESCRIPTOR_RELDIR);
    const bool carried
        = runsFromItsBuildTree() || installedShardArches(modelsRoot.parent_path()).count(arch) != 0;
    return !carried && !std::filesystem::exists(modelsRoot / "predict_engine" / arch);
}

/// Forward and backward graphs of [batch, heads, sequence, head_dim] = [4, 8, 1024, 128] in
/// bf16 with FLOAT compute, in the three dense layouts the models describe: token-major
/// (bshd), head-major (bhsd) and kv_bhsd, the decode KV cache (query and output token-major,
/// key and value head-major).
std::vector<std::pair<std::string, flatbuffers::FlatBufferBuilder>> sdpaLayoutGraphs()
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_test_sdk::utilities::createValidSdpaBwdGraph;
    using hipdnn_test_sdk::utilities::createValidSdpaFwdGraph;

    struct Layout
    {
        std::string name;
        std::vector<int64_t> queryAndOutput;
        std::vector<int64_t> keyAndValue;
    };
    const std::vector<int64_t> dims{4, 8, 1024, 128};
    const std::vector<int64_t> bshd{1048576, 128, 1024, 1};
    const std::vector<int64_t> bhsd{1048576, 131072, 128, 1};

    std::vector<std::pair<std::string, flatbuffers::FlatBufferBuilder>> graphs;
    for(const auto& [name, qo, kv] :
        {Layout{"bshd", bshd, bshd}, Layout{"bhsd", bhsd, bhsd}, Layout{"kv_bhsd", bshd, bhsd}})
    {
        graphs.emplace_back(
            "forward " + name,
            createValidSdpaFwdGraph(
                dims, qo, dims, kv, dims, kv, dims, qo, DataType::BFLOAT16, DataType::FLOAT));
        graphs.emplace_back(
            "backward " + name,
            mutateGraph(
                createValidSdpaBwdGraph(dims, qo, dims, kv, dims, kv, dims, qo, DataType::BFLOAT16),
                [](hipdnn_flatbuffers_sdk::data_objects::GraphT& graph) {
                    graph.nodes.front()->compute_data_type = DataType::FLOAT;
                }));
    }
    return graphs;
}

/// Every declared model must ship for its architecture, survive the loader and answer:
/// the engine itself gives an AVAILABLE `tflops` estimate for forward and backward graphs in
/// the token-major (bshd), head-major (bhsd) and kv_bhsd layouts, equal to that
/// architecture's own model's and below the device's peak, and describes the binding a model
/// must be trained against. Nothing else asks the engine for an estimate on a host, so a
/// model that never binds (an id typo, an unstaged file, a binding or architecture lookup
/// that drifted) would otherwise leave the engine unranked while every test passed. The plan
/// builder is mocked to serve each graph: which graphs the real builders serve depends on the
/// device, and is theirs to test (GetPredictionEstimatesEachLayoutOnTheDevice runs the real
/// ones). Outside the build tree a model may be absent only where the install carries no
/// shard for its architecture.
TEST_F(TestAsmSdpaEngine, EachDeclaredModelEstimatesOnItsArchitecture)
{
    namespace ingestor = hipdnn_plugin_sdk::ingestor;

    const auto graphs = sdpaLayoutGraphs();
    const EngineConfigWrapper config(nullptr, 0);
    size_t checked = 0;

    for(const auto& [arch, id] : AsmSdpaEngine::L1_MODEL_IDS)
    {
        const std::string archName(arch);
        SCOPED_TRACE(archName);
        if(modelMayBeAbsent(archName))
        {
            std::cout << "[ NOTE     ] this install carries no " << archName
                      << " content (its GPU family is packaged separately or has none), so the "
                      << archName << " ASM SDPA model is not checked here\n";
            continue;
        }
        ++checked;

        // The declared model alone, resolved from the catalog the engine binds from.
        const auto resolved = ingestor::resolveDeclaredEnginePredictions(
            hip_kernel_provider::kernel_ingestor_engine::descriptorCatalog(),
            AsmSdpaEngine::engineName(),
            AsmSdpaEngine::selectorRevision(),
            {{archName, {hipdnn_flatbuffers_sdk::utilities::parseUuid(id)}}});
        for(const auto& [metric, byArch] : resolved.refused)
        {
            for(const auto& [refusedArch, refusal] : byArch)
            {
                ADD_FAILURE() << "the loader refused " << id << " (" << metric << ", "
                              << refusedArch << "): " << refusal.reason;
            }
        }
        const auto tflops = resolved.byMetric.find("tflops");
        ASSERT_TRUE(tflops != resolved.byMetric.end() && tflops->second.count(archName) == 1)
            << "no tflops model " << id << " ships for " << archName;
        hipdnn_plugin_sdk::uhd::EngineModelBinding own;
        own.bind("tflops",
                 archName,
                 ingestor::UhdKernelHeuristic::configFrom(tflops->second.at(archName)));

        const auto device = trainingDevice(archName);
        SynthesizedDeviceEngine engine(device);
        engine.addPlanBuilder(servingPlanBuilder());

        for(const auto& [name, builder] : graphs)
        {
            SCOPED_TRACE(name);
            const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
            const auto estimate = engine.getPrediction(
                _handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true);
            ASSERT_EQ(estimate.status, PredictionStatus::AVAILABLE) << estimate.reason;
            EXPECT_EQ(estimate.metric, "tflops");
            EXPECT_EQ(estimate.engine_id, AsmSdpaEngine::staticId());
            EXPECT_EQ(estimate.kind, PredictionKind::ENGINE);

            const auto expected
                = own.predict(AsmSdpaEngine::staticId(),
                              AsmSdpaEngine::engineName(),
                              AsmSdpaEngine::selectorRevision(),
                              "tflops",
                              archName,
                              hipdnn_plugin_sdk::heuristics::engineFeatures(graph, config, device),
                              true);
            ASSERT_EQ(expected.status, PredictionStatus::AVAILABLE) << expected.reason;
            EXPECT_DOUBLE_EQ(estimate.value, expected.value)
                << "the engine did not answer with " << archName << "'s own model";
            EXPECT_TRUE(std::isfinite(estimate.value) && estimate.value > 0.0) << estimate.value;
            EXPECT_LT(estimate.value, denseBf16PeakTflops(archName))
                << "an estimate above the device's dense bf16 peak";
        }

        const GraphWrapper firstGraph(graphs.front().second.GetBufferPointer(),
                                      graphs.front().second.GetSize());

        // Described rather than evaluated: the binding a model must be trained against.
        const auto description = engine.getPrediction(
            _handle, firstGraph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, false);
        ASSERT_FALSE(description.binding_json.empty()) << description.reason;
        const auto binding = nlohmann::json::parse(description.binding_json);
        EXPECT_EQ(binding.at("trained_against").at("selector_revision").get<std::string>(),
                  AsmSdpaEngine::selectorRevision());
        EXPECT_EQ(binding.at("uhd_id").get<std::string>(), std::string(id));
        EXPECT_EQ(description.uhd_id, std::string(id));

        // Only `tflops` models ship: another metric is unanswered, not substituted.
        const auto timeConfigBuilder = engineConfigNaming("time");
        const auto inTime = engine.getPrediction(_handle,
                                                 firstGraph,
                                                 wrapConfig(timeConfigBuilder),
                                                 HIPDNN_ENGINE_PREDICTION_ENGINE,
                                                 true);
        EXPECT_EQ(inTime.status, PredictionStatus::UNAVAILABLE) << inTime.reason;
        EXPECT_EQ(inTime.metric, "time");
    }
    if(checked == 0)
    {
        GTEST_SKIP() << "this install carries content for neither architecture the ASM SDPA "
                        "models are declared for";
    }
}

/// A model that records another selector revision is refused at load, and the engine says so
/// by name instead of estimating from it. The engine binds from a catalog whose gfx942 model
/// carries a stale revision.
TEST_F(TestAsmSdpaEngine, AModelRecordingAnotherSelectorRevisionIsRefusedNamingBoth)
{
    namespace ingestor = hipdnn_plugin_sdk::ingestor;
    const auto& [arch, id] = AsmSdpaEngine::L1_MODEL_IDS.front();
    const std::string archName(arch);
    if(modelMayBeAbsent(archName))
    {
        GTEST_SKIP() << "this install carries no " << archName << " content";
    }

    ingestor::DescriptorCatalog stale;
    stale.heuristics = hip_kernel_provider::kernel_ingestor_engine::descriptorCatalog().heuristics;
    const auto model = stale.heuristics.find(hipdnn_flatbuffers_sdk::utilities::parseUuid(id));
    ASSERT_TRUE(model != stale.heuristics.end()) << "no model " << id << " in the catalog";
    const std::string staleRevision = "hip-kernel-provider/asm-sdpa/0000000000000000";
    ASSERT_NE(staleRevision, AsmSdpaEngine::selectorRevision());
    model->second.descriptor.trainedAgainstSelectorRevision = staleRevision;

    SynthesizedDeviceEngine engine(trainingDevice(archName), stale);
    engine.addPlanBuilder(servingPlanBuilder());
    auto graphBuilder = createSdpaFwdGraph();
    const GraphWrapper graph(graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const EngineConfigWrapper config(nullptr, 0);

    const auto estimate
        = engine.getPrediction(_handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true);
    EXPECT_EQ(estimate.status, PredictionStatus::UNAVAILABLE);
    EXPECT_THAT(estimate.reason, ::testing::HasSubstr(staleRevision));
    EXPECT_THAT(estimate.reason, ::testing::HasSubstr(AsmSdpaEngine::selectorRevision()));
}

/// What the handle's own device gets: the real handle, the real device lookup and the real
/// plan builders, on the architectures the models are declared for, in each layout they
/// describe. Every other test reaches an estimate through a synthesized device and a mocked
/// builder, so this is the one that runs the production path end to end.
TEST_F(TestAsmSdpaEngine, GetPredictionEstimatesEachLayoutOnTheDevice)
{
    SKIP_IF_NO_DEVICES();

    const auto arch = hip_kernel_provider_common::getDeviceString(_handle.getStream());
    if(arch != "gfx942" && arch != "gfx950")
    {
        GTEST_SKIP();
    }
    if(modelMayBeAbsent(arch))
    {
        GTEST_SKIP() << "this install carries no " << arch << " content";
    }

    const EngineConfigWrapper config(nullptr, 0);
    for(const auto& [name, builder] : sdpaLayoutGraphs())
    {
        SCOPED_TRACE(name);
        const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());
        const auto estimate
            = _engine.getPrediction(_handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true);
        ASSERT_EQ(estimate.status, PredictionStatus::AVAILABLE) << estimate.reason;
        EXPECT_EQ(estimate.metric, "tflops");
        EXPECT_EQ(estimate.engine_id, AsmSdpaEngine::staticId());
        EXPECT_TRUE(std::isfinite(estimate.value) && estimate.value > 0.0) << estimate.value;
        EXPECT_LT(estimate.value, denseBf16PeakTflops(arch))
            << "an estimate above the device's dense bf16 peak";
    }
}
#endif // HIPKERNELPROVIDER_ASM_SDPA_DESCRIPTOR_RELDIR
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

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

/// A plan that needs more workspace than global.workspace_size_limit allows is refused with
/// INVALID_VALUE by both entry points and is never built; a plan at the limit, or with no
/// limit set, is not. The status is the one the SDK's ingestor engines raise for the same
/// condition. It is a refused plan, not an applicability decline: the limit belongs to the
/// engine configuration, which applicability never sees.
TEST_F(TestAsmSdpaEngine, APlanOverTheWorkspaceLimitIsRefusedAndNeverBuilt)
{
    constexpr size_t PLAN_BYTES = 1024;
    auto builder = std::make_unique<::testing::NiceMock<MockPlanBuilder>>();
    ON_CALL(*builder, isApplicable).WillByDefault(::testing::Return(true));
    ON_CALL(*builder, getMaxWorkspaceSize).WillByDefault(::testing::Return(PLAN_BYTES));
    int plansBuilt = 0;
    ON_CALL(*builder, buildPlan).WillByDefault([&](auto&&...) { ++plansBuilt; });
    AsmSdpaEngine engine;
    engine.addPlanBuilder(std::move(builder));
    const hipdnn_test_sdk::utilities::MockGraph graph;

    const auto under
        = createIntKnobConfig(hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, PLAN_BYTES - 1);
    Context refused;
    EXPECT_EQ(statusOf([&] {
                  static_cast<void>(engine.getMaxWorkspaceSize(_handle, graph, wrapConfig(under)));
              }),
              HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_EQ(statusOf([&] {
                  engine.initializeExecutionContext(_handle, graph, wrapConfig(under), refused);
              }),
              HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
    EXPECT_EQ(plansBuilt, 0);

    const auto exact
        = createIntKnobConfig(hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, PLAN_BYTES);
    EXPECT_EQ(engine.getMaxWorkspaceSize(_handle, graph, wrapConfig(exact)), PLAN_BYTES);
    Context atLimit;
    engine.initializeExecutionContext(_handle, graph, wrapConfig(exact), atLimit);
    EXPECT_EQ(plansBuilt, 1);

    const auto unlimited = createEngineConfigWithoutKnobs();
    EXPECT_EQ(engine.getMaxWorkspaceSize(_handle, graph, wrapConfig(unlimited)), PLAN_BYTES);
}

/// A negative limit is rejected whatever the plan needs: it is a bad value, not a limit.
TEST_F(TestAsmSdpaEngine, ANegativeWorkspaceLimitIsAnInvalidValue)
{
    auto builder = std::make_unique<::testing::NiceMock<MockPlanBuilder>>();
    ON_CALL(*builder, isApplicable).WillByDefault(::testing::Return(true));
    AsmSdpaEngine engine;
    engine.addPlanBuilder(std::move(builder));
    const hipdnn_test_sdk::utilities::MockGraph graph;

    const auto negative
        = createIntKnobConfig(hipdnn_plugin_sdk::WORKSPACE_SIZE_LIMIT_KNOB_NAME, -1);
    EXPECT_EQ(statusOf([&] {
                  static_cast<void>(
                      engine.getMaxWorkspaceSize(_handle, graph, wrapConfig(negative)));
              }),
              HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
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
        {createAccumulatorKnobConfig("a64"), createIntKnobConfig(K_ACC_KNOB, 16)})
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

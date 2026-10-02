// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericEngine.hpp>

#include "KernelIngestorTestFixtures.hpp"

/**
 * @file TestGenericEngine.cpp
 * @brief Unit tests for GenericEngine.hpp: knob validation at construction and
 *        IEngine overrides delegating to the plan builder and state manager.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_plugin_sdk::ingestor::testing;

using StubEngine = GenericEngine<StubHandle, StubSettings, StubContext>;

static_assert(!std::is_move_constructible_v<StubEngine>);
static_assert(!std::is_move_assignable_v<StubEngine>);
static_assert(!std::is_copy_constructible_v<StubEngine>);
static_assert(!std::is_copy_assignable_v<StubEngine>);

TEST(TestIngestorGenericEngine, AcceptsAKnobNamingADeclaredMetadataField)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;

    EXPECT_NO_THROW(
        (StubEngine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver)));
}

TEST(TestIngestorGenericEngine, RejectsAKnobNamingNoMetadataField)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;

    EXPECT_THROW(
        (StubEngine(makeEngineWithKnobs({"no_such_field"}), makeStubStateManager(), resolver)),
        std::invalid_argument);
}

TEST(TestIngestorGenericEngine, IdHashesTheUedNameIntoHipdnnsEngineIdSpace)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    EXPECT_NE(engine.id(), 0);
}

TEST(TestIngestorGenericEngine, IsApplicableTrueWhenTheStateManagerHasASurvivingKernel)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    StubHandle handle;
    const TestGraph graph(makeGraphId(0x60));

    EXPECT_TRUE(engine.isApplicable(handle, graph));
}

TEST(TestIngestorGenericEngine, IsApplicableFalseWhenNoMatcherAccepts)
{
    // Distinct symbol avoids colliding with ScopedTestSymbols' graph match elsewhere.
    constexpr const char* REJECT_SYMBOL = "hipdnn.kernel_ingestor.test.generic_engine.reject";
    GraphMatchRegistry::registerSymbol(REJECT_SYMBOL, &rejectGraph);
    const ScopedBlockSizeScore scorer;

    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.name = "test schema";
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};

    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.name = "test pack";
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT")};

    auto stateManager = std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::make_shared<NativeKernelHeuristic>(SCORE_SYMBOL),
        REJECT_SYMBOL);

    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), std::move(stateManager), resolver);

    StubHandle handle;
    const TestGraph graph(makeGraphId(0x61));

    EXPECT_FALSE(engine.isApplicable(handle, graph));

    GraphMatchRegistry::unregisterSymbol(REJECT_SYMBOL);
}

TEST(TestIngestorGenericEngine, GetDetailsReportsTheEnginesKnobs)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    StubHandle handle;
    const TestGraph graph(makeGraphId(0x62));
    hipdnnPluginConstData_t details{};

    engine.getDetails(handle, graph, details);

    ASSERT_NE(details.ptr, nullptr);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(details.ptr,
                                                                                     details.size);
    ASSERT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.engineId(), engine.id());
    // The UED name, so a graph-time record identifies its engine the same way the
    // getEngineName entry point does.
    EXPECT_EQ(wrapper.name(), "test:engine");
    // GenericEngine::getDetails() always prepends the out-of-band benchmarking knob
    // (Task 1.4), so a UED declaring one knob of its own advertises two; looked up by
    // name, since the prepend fixes a position Phase 2 must not assume by index either.
    ASSERT_EQ(wrapper.knobCount(), 2U);
    EXPECT_EQ(wrapper.getKnobByName(BLOCK_SIZE).knobId(), BLOCK_SIZE);
}

/// GenericEngine::getDetails() advertises global.benchmarking out-of-band, so a UED
/// declaring zero knobs of its own still reports exactly this one knob -- and its
/// value semantics (int, default 0, min/max 0/1) match MIOpen's createBenchmarkingKnob
/// (plan design record, Finding 1). Looked up by name: the prepend fixes a position
/// no test should assume by index.
TEST(TestIngestorGenericEngine, GetDetailsAdvertisesTheBenchmarkingKnobOutOfBand)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    StubHandle handle;
    const TestGraph graph(makeGraphId(0x65));
    hipdnnPluginConstData_t details{};

    engine.getDetails(handle, graph, details);

    ASSERT_NE(details.ptr, nullptr);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(details.ptr,
                                                                                     details.size);
    ASSERT_TRUE(wrapper.isValid());

    const auto& knob = wrapper.getKnobByName(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME);
    EXPECT_EQ(knob.knobId(), hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME);

    ASSERT_TRUE(knob.hasDefaultValue());
    EXPECT_EQ(knob.defaultValueType(), hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue);
    const auto& defaultValue
        = knob.defaultValueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>();
    EXPECT_EQ(defaultValue.value(), 0);

    ASSERT_TRUE(knob.hasConstraint());
    EXPECT_EQ(knob.constraintType(),
              hipdnn_flatbuffers_sdk::data_objects::KnobConstraint::IntConstraint);
    const auto& constraint
        = knob.constraintAs<hipdnn_flatbuffers_sdk::data_objects::IntConstraint>();
    EXPECT_EQ(constraint.min_value(), 0);
    EXPECT_EQ(constraint.max_value(), 1);
    EXPECT_EQ(constraint.step(), 1);
}

/// A UED naming no knobs of its own still gets the out-of-band prepend: advertisement
/// does not depend on the engine declaring anything.
TEST(TestIngestorGenericEngine, GetDetailsAdvertisesExactlyTheBenchmarkingKnobWhenNoneAreDeclared)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubEngine engine(makeEngineWithKnobs({}), makeStubStateManager(), resolver);

    StubHandle handle;
    const TestGraph graph(makeGraphId(0x66));
    hipdnnPluginConstData_t details{};

    engine.getDetails(handle, graph, details);

    ASSERT_NE(details.ptr, nullptr);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(details.ptr,
                                                                                     details.size);
    ASSERT_TRUE(wrapper.isValid());
    ASSERT_EQ(wrapper.knobCount(), 1U);
    EXPECT_EQ(wrapper.getKnobByName(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME).knobId(),
              hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME);
}

/// The out-of-band knob never enters EngineDescriptor.knobs, so a UED declaring no
/// knobs must not trip findUndeclaredKnob's std::invalid_argument.
TEST(TestIngestorGenericEngine, ConstructingAnEngineWithNoDeclaredKnobsNeverThrows)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;

    EXPECT_NO_THROW((StubEngine(makeEngineWithKnobs({}), makeStubStateManager(), resolver)));
}

TEST(TestIngestorGenericEngine, GetMaxWorkspaceSizeDelegatesToThePlanBuilder)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    const StubHandle handle;
    const TestGraph graph(makeGraphId(0x63));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper emptyConfig(nullptr, 0);

    // Stub manager ships one 64-block kernel; only the plan builder having run
    // explains this value.
    EXPECT_EQ(engine.getMaxWorkspaceSize(handle, graph, emptyConfig), 64U);
}

TEST(TestIngestorGenericEngine, InitializeExecutionContextDelegatesToThePlanBuilder)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);

    const StubHandle handle;
    const TestGraph graph(makeGraphId(0x64));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper emptyConfig(nullptr, 0);
    StubContext context;

    engine.initializeExecutionContext(handle, graph, emptyConfig, context);

    EXPECT_TRUE(context.hasPlan());
}

TEST(TestIngestorGenericEngine, BrokenEngineModelDoesNotRemoveGraphApplicability)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    auto descriptor = makeEngineWithKnobs({BLOCK_SIZE});
    HeuristicDescriptor model;
    model.id = testId(0xA4);
    model.name = "invalid kernel-dependent L1 model";
    model.adapter = UhdAdapter::NATIVE;
    model.nativeSymbol = "missing.l1.scorer";
    model.featuresSignature = {"$kernel.block_size"};
    model.score = {"tflops", true, "identity"};
    model.engineName = descriptor.name;
    model.role = "predict_engine";
    model.arch = "default";
    // Provenance matches, so only the kernel feature refuses this model.
    model.trainedAgainstSelectorRevision = "selector-test";
    model.trainedAgainstJson
        = {{"ued", {{"id", "20112233-4455-6677-8899-aabbccddeeff"}, {"revision", "1.0"}}},
           {"kmd", {{"id", "30112233-4455-6677-8899-aabbccddeeff"}, {"revision", "1.0"}}},
           {"umd", nlohmann::json::array()}};
    const StubEngine engine(std::move(descriptor),
                            makeStubStateManager(),
                            resolver,
                            {{"tflops", {{"default", model}}}},
                            {},
                            "selector-test");
    StubHandle handle;
    const TestGraph graph(makeGraphId(0x67));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(nullptr, 0);
    const auto described
        = engine.getPrediction(handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, false);
    EXPECT_EQ(nlohmann::json::parse(described.features_json).at("device.cu_count"), 304);
    EXPECT_FALSE(nlohmann::json::parse(described.features_json).contains("graph.flops"));
    const auto evaluated
        = engine.getPrediction(handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true);
    EXPECT_EQ(evaluated.status, PredictionStatus::INVALID);
    EXPECT_EQ(evaluated.metric, "tflops") << "a config naming no metric asks in the default";
    EXPECT_TRUE(engine.isApplicable(handle, graph));
}

/// The L1 test scorer: its single feature, `device.cu_count` (304 on the stub device).
double firstL1Feature(const double* values, size_t count)
{
    return count == 0 ? 0.0 : values[0];
}

/// A descriptor-backed engine's `predict_engine` model answers only for the selector revision
/// it was trained against, as on the opaque-engine path.
TEST(TestIngestorGenericEngine, AnEngineModelAnswersOnlyForTheSelectorItWasTrainedAgainst)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    constexpr const char* L1_SYMBOL = "hipdnn.kernel_ingestor.test.generic_engine.l1_cu_count";
    hipdnn_plugin_sdk::uhd::NativeScorerRegistry::registerSymbol(L1_SYMBOL, firstL1Feature);
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const std::vector<nlohmann::json> signature = {"$device.cu_count"};
    HeuristicDescriptor model;
    model.id = testId(0xA5);
    model.name = "engine throughput";
    model.adapter = UhdAdapter::NATIVE;
    model.nativeSymbol = L1_SYMBOL;
    model.featuresSignature = signature;
    model.featuresHash = hipdnn_plugin_sdk::uhd::FeatureExtractor::computeHash(signature);
    model.objective = "max";
    model.score = {"tflops", true, "identity"};
    const auto evaluate = [&](const std::string& trainedAgainst) {
        model.trainedAgainstSelectorRevision = trainedAgainst;
        model.trainedAgainstJson
            = {{"ued", {{"id", "20112233-4455-6677-8899-aabbccddeeff"}, {"revision", "1.0"}}},
               {"kmd", {{"id", "30112233-4455-6677-8899-aabbccddeeff"}, {"revision", "1.0"}}},
               {"umd", nlohmann::json::array()}};
        if(!trainedAgainst.empty())
        {
            model.trainedAgainstJson["selector_revision"] = trainedAgainst;
        }
        const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}),
                                makeStubStateManager(),
                                resolver,
                                {{"tflops", {{"default", model}}}},
                                {},
                                "selector-new");
        StubHandle handle;
        const TestGraph graph(makeGraphId(0x6C));
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(nullptr, 0);
        return engine.getPrediction(handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, true);
    };

    const auto current = evaluate("selector-new");
    EXPECT_EQ(current.status, PredictionStatus::AVAILABLE) << current.reason;
    EXPECT_DOUBLE_EQ(current.value, 304.0);

    // Wrong build, not a bad model: UNAVAILABLE, naming both revisions.
    const auto stale = evaluate("selector-old");
    EXPECT_EQ(stale.status, PredictionStatus::UNAVAILABLE);
    EXPECT_NE(stale.reason.find("selector-old"), std::string::npos) << stale.reason;
    EXPECT_NE(stale.reason.find("selector-new"), std::string::npos) << stale.reason;
    EXPECT_DOUBLE_EQ(stale.value, 0.0);

    // Nothing to compare against is a contract failure, as on the opaque-engine path.
    const auto unrecorded = evaluate("");
    EXPECT_EQ(unrecorded.status, PredictionStatus::INVALID);
    EXPECT_NE(unrecorded.reason.find("selector_revision"), std::string::npos) << unrecorded.reason;

    hipdnn_plugin_sdk::uhd::NativeScorerRegistry::unregisterSymbol(L1_SYMBOL);
}

/// An engine config as the backend stamps one: no knobs, only the ranking metric.
flatbuffers::FlatBufferBuilder configWithMetric(const std::string& metric)
{
    hipdnn_flatbuffers_sdk::data_objects::EngineConfigT config;
    config.ranking_metric = metric;
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::EngineConfig::Pack(builder, &config));
    return builder;
}

/// RFC 0019 §4.4: an unregistered metric has no direction to rank by. BAD_PARAM rather than
/// UNAVAILABLE, so a typo does not read as "no engine serves this".
TEST(TestIngestorGenericEngine, AnUnregisteredRankingMetricIsABadParameter)
{
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), makeStubStateManager(), resolver);
    StubHandle handle;
    const TestGraph graph(makeGraphId(0x69));
    const auto buffer = configWithMetric("latency");
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
        buffer.GetBufferPointer(), buffer.GetSize());

    const auto expectBadParam = [](const auto& call) {
        try
        {
            call();
            ADD_FAILURE() << "an unregistered metric was accepted";
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
        {
            EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
        }
    };
    for(const auto kind : {HIPDNN_ENGINE_PREDICTION_ENGINE, HIPDNN_ENGINE_PREDICTION_CONFIGURATION})
    {
        expectBadParam([&] { engine.getPrediction(handle, graph, config, kind, true); });
    }
    // Plan build ranks by the same field, so it refuses the same way.
    StubContext context;
    expectBadParam([&] { engine.initializeExecutionContext(handle, graph, config, context); });
}

enum class ConfigurationCatalog
{
    SINGLETON,
    DISTINCT_KNOBS,
    AMBIGUOUS_KNOBS
};

class TestIngestorConfigurationPrediction : public ::testing::TestWithParam<ConfigurationCatalog>
{
};

TEST_P(TestIngestorConfigurationPrediction, ReturnsOnlyUniquelyAddressableConfigurations)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};
    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT")};
    if(GetParam() != ConfigurationCatalog::SINGLETON)
    {
        // Distinct metadata tuples may still share every exposed integer knob.
        pack.kernels.push_back(
            makeTestKernel(testId(0x65),
                           "other_kernel",
                           GetParam() == ConfigurationCatalog::DISTINCT_KNOBS ? 128 : 64,
                           GetParam() == ConfigurationCatalog::AMBIGUOUS_KNOBS ? "HALF" : "FLOAT"));
    }
    HeuristicDescriptor model;
    model.id = HEURISTIC_ID;
    model.adapter = UhdAdapter::NATIVE;
    model.nativeSymbol = SCORE_SYMBOL;
    model.score = {"tflops", true, "identity"};
    auto ranker = UhdKernelHeuristic::tryCreate(model, "calibrated configuration", {BLOCK_SIZE});
    ASSERT_NE(ranker, nullptr);
    auto manager = std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::move(ranker),
        GRAPH_MATCH_SYMBOL);
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), std::move(manager), resolver);
    StubHandle handle;
    const TestGraph graph(makeGraphId(0x68));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(nullptr, 0);
    const auto prediction
        = engine.getPrediction(handle, graph, config, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);
    if(GetParam() == ConfigurationCatalog::AMBIGUOUS_KNOBS)
    {
        EXPECT_EQ(prediction.status, PredictionStatus::UNAVAILABLE);
        EXPECT_EQ(prediction.engine_config, nullptr);
        return;
    }
    ASSERT_EQ(prediction.status, PredictionStatus::AVAILABLE);
    const auto expectedBlockSize = GetParam() == ConfigurationCatalog::SINGLETON ? 64 : 128;
    EXPECT_EQ(prediction.metric, "tflops");
    EXPECT_DOUBLE_EQ(prediction.value, expectedBlockSize);
    ASSERT_NE(prediction.engine_config, nullptr);
    flatbuffers::FlatBufferBuilder serialized;
    serialized.Finish(EngineConfig::Pack(serialized, prediction.engine_config.get()));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper selected(
        serialized.GetBufferPointer(), serialized.GetSize());
    EXPECT_EQ(engine.getMaxWorkspaceSize(handle, graph, selected), expectedBlockSize);
    hipdnnPluginConstData_t details{};
    engine.enumerateCandidates(handle, graph, selected, 0, 10, details);
    const auto* candidates = GetEngineDetails(details.ptr)->candidate_page();
    ASSERT_NE(candidates, nullptr);
    EXPECT_EQ(candidates->total_count(), 1U);
    ASSERT_EQ(candidates->candidates()->size(), 1U);
    EXPECT_EQ(candidates->candidates()->Get(0)->id()->str(),
              toString(testId(GetParam() == ConfigurationCatalog::SINGLETON ? 0x64 : 0x65)));

    // Asked in `time`, L2 has no ranker of that metric and must not report the tflops value as
    // a time (RFC 0019 §11.4).
    const auto timeBuffer = configWithMetric("time");
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper timeConfig(
        timeBuffer.GetBufferPointer(), timeBuffer.GetSize());
    const auto inTime = engine.getPrediction(
        handle, graph, timeConfig, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);
    EXPECT_EQ(inTime.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(inTime.metric, "time");
    EXPECT_EQ(inTime.engine_config, nullptr);
}

INSTANTIATE_TEST_SUITE_P(KnobTuples,
                         TestIngestorConfigurationPrediction,
                         ::testing::Values(ConfigurationCatalog::SINGLETON,
                                           ConfigurationCatalog::DISTINCT_KNOBS,
                                           ConfigurationCatalog::AMBIGUOUS_KNOBS));

/// RFC 0019 §5 step 9: a benchmark record covering the catalog decides the predicted
/// configuration and its value. The model prefers K128; the record measured K64 faster.
TEST(TestIngestorGenericEngine, ACoveringRecordDecidesTheConfigurationPredictionAndItsValue)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};
    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeTestKernel(testId(0x65), "kernel_128_float", 128, "FLOAT")};
    HeuristicDescriptor model;
    model.id = HEURISTIC_ID;
    model.adapter = UhdAdapter::NATIVE;
    model.nativeSymbol = SCORE_SYMBOL; // the block size: K128 scores 128 tflops, K64 64
    model.score = {"tflops", true, "identity"};
    auto ranker = UhdKernelHeuristic::tryCreate(model, "calibrated configuration", {BLOCK_SIZE});
    ASSERT_NE(ranker, nullptr);
    auto owned = std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::move(ranker),
        GRAPH_MATCH_SYMBOL);
    auto& manager = *owned;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), std::move(owned), resolver);
    StubHandle handle;
    const auto properties = testDeviceProperties();

    // K64 measured at 1 ms, K128 at 2 ms, covering the whole catalog.
    const auto measure = [&](const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) {
        WinnerRecord record;
        for(const auto& kernel : manager.unsortedDefinitions(MatchContext{graph, 0, properties}))
        {
            record.push_back({kernel.kernelId,
                              kernel.packId,
                              kernel.dispatchId,
                              kernel.getIntMetadata(BLOCK_SIZE) == 64 ? 1.0 : 2.0});
        }
        std::sort(record.begin(), record.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.timeMs < rhs.timeMs;
        });
        ASSERT_EQ(record.size(), 2U);
        manager.recordWinner(
            WinnerKey{hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey{graph},
                      DeviceKey{properties}},
            record,
            WinnerWriteCause::FRESH_MISS);
    };
    const auto predict = [&](const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                             const std::string& metric) {
        const auto buffer = configWithMetric(metric);
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
            buffer.GetBufferPointer(), buffer.GetSize());
        return engine.getPrediction(
            handle, graph, config, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);
    };
    // The kernel a predicted configuration replays to.
    const auto pinned = [&](const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                            const EnginePredictionT& prediction) {
        flatbuffers::FlatBufferBuilder serialized;
        serialized.Finish(EngineConfig::Pack(serialized, prediction.engine_config.get()));
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper selected(
            serialized.GetBufferPointer(), serialized.GetSize());
        hipdnnPluginConstData_t details{};
        engine.enumerateCandidates(handle, graph, selected, 0, 10, details);
        const auto* candidates = GetEngineDetails(details.ptr)->candidate_page()->candidates();
        return candidates->size() == 1 ? candidates->Get(0)->id()->str() : std::string();
    };

    // The record measures time, so it answers without a model.
    const TestGraph timed(makeGraphId(0x6D));
    measure(timed);
    const auto inTime = predict(timed, "time");
    ASSERT_EQ(inTime.status, PredictionStatus::AVAILABLE) << inTime.reason;
    ASSERT_NE(inTime.engine_config, nullptr);
    EXPECT_EQ(pinned(timed, inTime), toString(testId(0x64)));
    EXPECT_DOUBLE_EQ(inTime.value, 1.0);
    EXPECT_TRUE(inTime.uhd_id.empty()) << "a measured value was attributed to a model";

    // Throughput derives from the measured time and the graph's work: 2*1024^3 flops in 1 ms.
    const MatmulTestGraph matmul(1024, 1024, 1024);
    measure(matmul.graph());
    const auto inTflops = predict(matmul.graph(), "tflops");
    ASSERT_EQ(inTflops.status, PredictionStatus::AVAILABLE) << inTflops.reason;
    ASSERT_NE(inTflops.engine_config, nullptr);
    EXPECT_EQ(pinned(matmul.graph(), inTflops), toString(testId(0x64)));
    EXPECT_DOUBLE_EQ(inTflops.value, 2.0 * 1024 * 1024 * 1024 / 1e9);
    EXPECT_TRUE(inTflops.uhd_id.empty());

    // With no published work a time cannot become a throughput: the record still picks the
    // configuration and the model estimates its value.
    const TestGraph unknownWork(makeGraphId(0x6E));
    measure(unknownWork);
    const auto estimated = predict(unknownWork, "tflops");
    ASSERT_EQ(estimated.status, PredictionStatus::AVAILABLE) << estimated.reason;
    ASSERT_NE(estimated.engine_config, nullptr);
    EXPECT_EQ(pinned(unknownWork, estimated), toString(testId(0x64)));
    EXPECT_DOUBLE_EQ(estimated.value, 64.0);
    EXPECT_EQ(estimated.uhd_id, toString(HEURISTIC_ID));
}

/// Plan build and configuration prediction must read one measured snapshot: the catalog cache
/// keeps a measured order after the winner cache (capacity 1 here) evicts its record.
TEST(TestIngestorGenericEngine, AnEvictedRecordStillDecidesBothPlanAndPrediction)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler; // sizes workspace by block size: names the built kernel
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};
    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeTestKernel(testId(0x65), "kernel_128_float", 128, "FLOAT")};
    HeuristicDescriptor model;
    model.id = HEURISTIC_ID;
    model.adapter = UhdAdapter::NATIVE;
    model.nativeSymbol = SCORE_SYMBOL; // the block size: the model prefers K128
    model.score = {"tflops", true, "identity"};
    auto ranker = UhdKernelHeuristic::tryCreate(model, "calibrated configuration", {BLOCK_SIZE});
    ASSERT_NE(ranker, nullptr);
    auto owned = std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::move(ranker),
        GRAPH_MATCH_SYMBOL,
        "",
        KernelIngestorStateManager<StubHandle>::DEFAULT_CATALOG_CACHE_CAPACITY,
        EngineIdentity{},
        /*winnerCacheCapacity=*/1);
    auto& manager = *owned;
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), std::move(owned), resolver);
    StubHandle handle;
    const auto properties = testDeviceProperties();
    const auto keyFor = [&](const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) {
        return WinnerKey{hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey{graph},
                         DeviceKey{properties}};
    };
    // Identified, so its catalog is cached; real work, so a time converts to a throughput.
    const MatmulTestGraph matmul(1024, 1024, 1024, makeGraphId(0x6F));
    const auto& graph = matmul.graph();

    // K64 measured at 1 ms, K128 at 2 ms, covering the whole catalog.
    WinnerRecord record;
    for(const auto& kernel : manager.unsortedDefinitions(MatchContext{graph, 0, properties}))
    {
        record.push_back({kernel.kernelId,
                          kernel.packId,
                          kernel.dispatchId,
                          kernel.getIntMetadata(BLOCK_SIZE) == 64 ? 1.0 : 2.0});
    }
    std::sort(record.begin(), record.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timeMs < rhs.timeMs;
    });
    ASSERT_EQ(record.size(), 2U);
    manager.recordWinner(keyFor(graph), record, WinnerWriteCause::FRESH_MISS);

    const auto buffer = configWithMetric("tflops");
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
        buffer.GetBufferPointer(), buffer.GetSize());
    const auto builtBlockSize
        = [&](const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) {
              StubContext context;
              engine.initializeExecutionContext(handle, graph, engineConfig, context);
              return context.plan().getWorkspaceSize(handle);
          };
    // Plan build adopts the record and caches the measured catalog.
    ASSERT_EQ(builtBlockSize(config), 64U);

    const MatmulTestGraph other(512, 512, 512);
    manager.recordWinner(keyFor(other.graph()), record, WinnerWriteCause::FRESH_MISS);
    ASSERT_FALSE(manager.winnerFor(keyFor(graph)).has_value())
        << "the precondition: the record that ordered the cached catalog is evicted";

    const auto prediction
        = engine.getPrediction(handle, graph, config, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);
    ASSERT_EQ(prediction.status, PredictionStatus::AVAILABLE) << prediction.reason;
    ASSERT_NE(prediction.engine_config, nullptr);
    EXPECT_DOUBLE_EQ(prediction.value, 2.0 * 1024 * 1024 * 1024 / 1e9)
        << "the measured K64 throughput, not the model's estimate";
    EXPECT_TRUE(prediction.uhd_id.empty()) << "a measured value was attributed to a model";

    // The predicted configuration builds the same kernel as the unpinned plan.
    flatbuffers::FlatBufferBuilder serialized;
    serialized.Finish(EngineConfig::Pack(serialized, prediction.engine_config.get()));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper predicted(
        serialized.GetBufferPointer(), serialized.GetSize());
    EXPECT_EQ(builtBlockSize(config), 64U);
    EXPECT_EQ(builtBlockSize(predicted), 64U);
}

/// The uniqueness check must compare non-integer knobs by ordinal, so a string knob can be the
/// only thing that distinguishes two kernels.
TEST(TestIngestorGenericEngine, AConfigurationDistinguishedByAStringKnobIsPredictable)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    const ScopedTestSymbols symbols;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};
    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    // Equal block sizes: only the string knob addresses either kernel.
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeTestKernel(testId(0x65), "kernel_64_half", 64, "HALF")};
    HeuristicDescriptor model;
    model.id = HEURISTIC_ID;
    model.adapter = UhdAdapter::NATIVE;
    model.nativeSymbol = SCORE_SYMBOL;
    model.score = {"tflops", true, "identity"};
    auto ranker
        = UhdKernelHeuristic::tryCreate(model, "calibrated configuration", {BLOCK_SIZE, DTYPE});
    ASSERT_NE(ranker, nullptr);
    auto manager = std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::move(ranker),
        GRAPH_MATCH_SYMBOL);
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE, DTYPE}), std::move(manager), resolver);
    StubHandle handle;
    const TestGraph graph(makeGraphId(0x6A));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(nullptr, 0);

    const auto prediction
        = engine.getPrediction(handle, graph, config, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, true);

    ASSERT_EQ(prediction.status, PredictionStatus::AVAILABLE) << prediction.reason;
    ASSERT_NE(prediction.engine_config, nullptr);
    // Replaying the predicted knobs (the string as its ordinal) enumerates exactly one kernel.
    flatbuffers::FlatBufferBuilder serialized;
    serialized.Finish(EngineConfig::Pack(serialized, prediction.engine_config.get()));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper selected(
        serialized.GetBufferPointer(), serialized.GetSize());
    hipdnnPluginConstData_t details{};
    engine.enumerateCandidates(handle, graph, selected, 0, 10, details);
    const auto* candidates = GetEngineDetails(details.ptr)->candidate_page();
    ASSERT_NE(candidates, nullptr);
    EXPECT_EQ(candidates->total_count(), 1U);
}

/// Counts every kernel a native scorer is asked about, so a test can prove a path ranks nothing.
constexpr const char* COUNTING_BLOCK_SIZE_SYMBOL
    = "hipdnn.kernel_ingestor.test.generic_engine.counting_block_size";
std::atomic<int> scoredKernels{0};

class ScopedCountingScorer
{
public:
    ScopedCountingScorer()
    {
        scoredKernels = 0;
        ScoreRegistry::registerSymbol(
            COUNTING_BLOCK_SIZE_SYMBOL,
            +[](const MatchContext&, const BoundTokens&, const KernelDefinition& kernel) {
                ++scoredKernels;
                return static_cast<double>(kernel.getIntMetadata(BLOCK_SIZE));
            });
    }
    ~ScopedCountingScorer()
    {
        ScoreRegistry::unregisterSymbol(COUNTING_BLOCK_SIZE_SYMBOL);
    }
    ScopedCountingScorer(const ScopedCountingScorer&) = delete;
    ScopedCountingScorer& operator=(const ScopedCountingScorer&) = delete;
};

/// A CONFIGURATION description names the calibrated ranker an evaluation would use, so
/// getPredictionCapabilities can list L2 models, and ranks nothing to do it.
TEST(TestIngestorGenericEngine, AConfigurationDescriptionNamesTheL2ModelWithoutRanking)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    const ScopedTestSymbols symbols;
    const ScopedCountingScorer scorer;
    const StubDeviceResolver resolver;
    const StubWorkspaceHandler handler;
    const ScopedDispatchRegistration<StubHandle> dispatch("hipdnn.kernel_ingestor.test.dispatch",
                                                          handler);
    MetadataSchema schema;
    schema.id = SCHEMA_ID;
    schema.fields = {{BLOCK_SIZE, MetadataType::INT, MetadataValue{int64_t{64}}},
                     {DTYPE, MetadataType::STRING, std::nullopt}};
    KernelDescriptorPack pack;
    pack.id = PACK_ID;
    pack.engineId = ENGINE_ID;
    pack.dispatchId = DISPATCH_ID;
    pack.kernels = {makeTestKernel(testId(0x64), "kernel_64_float", 64, "FLOAT"),
                    makeTestKernel(testId(0x65), "kernel_128_float", 128, "FLOAT")};
    HeuristicDescriptor model;
    model.id = HEURISTIC_ID;
    model.adapter = UhdAdapter::NATIVE;
    model.nativeSymbol = COUNTING_BLOCK_SIZE_SYMBOL;
    model.score = {"tflops", true, "identity"};
    // As the loader binds a metric's ranker: per metric, then architecture key.
    auto ranker = UhdKernelHeuristic::makeResolver(
        {{"tflops", {{"default", model}}}}, "engine 'test:engine'", {BLOCK_SIZE});
    auto manager = std::make_unique<KernelIngestorStateManager<StubHandle>>(
        std::move(schema),
        std::vector<MatchDescriptor>{},
        makeStubDispatches(),
        std::vector<KernelDescriptorPack>{std::move(pack)},
        std::move(ranker),
        GRAPH_MATCH_SYMBOL);
    const StubEngine engine(makeEngineWithKnobs({BLOCK_SIZE}), std::move(manager), resolver);
    StubHandle handle;
    const TestGraph graph(makeGraphId(0x6B));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(nullptr, 0);

    const auto described = engine.getPrediction(
        handle, graph, config, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, /*evaluate=*/false);
    EXPECT_EQ(described.status, PredictionStatus::UNAVAILABLE);
    EXPECT_EQ(described.uhd_id, toString(HEURISTIC_ID));
    const auto binding = nlohmann::json::parse(described.binding_json);
    EXPECT_EQ(binding.at("role"), "sort_kernel_catalog");
    EXPECT_EQ(binding.at("uhd_id"), toString(HEURISTIC_ID));

    // The only calibrated ranker estimates tflops, so a time description names no model.
    const auto timeBuffer = configWithMetric("time");
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper timeConfig(
        timeBuffer.GetBufferPointer(), timeBuffer.GetSize());
    const auto inTime = engine.getPrediction(
        handle, graph, timeConfig, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, /*evaluate=*/false);
    EXPECT_TRUE(inTime.uhd_id.empty());
    EXPECT_FALSE(nlohmann::json::parse(inTime.binding_json).contains("uhd_id"));

    (void)engine.getPrediction(
        handle, graph, config, HIPDNN_ENGINE_PREDICTION_ENGINE, /*evaluate=*/false);
    EXPECT_EQ(scoredKernels.load(), 0) << "describing a prediction ranked the catalog";

    // The counter is live: evaluating does rank, and names the same model.
    const auto evaluated = engine.getPrediction(
        handle, graph, config, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, /*evaluate=*/true);
    ASSERT_EQ(evaluated.status, PredictionStatus::AVAILABLE) << evaluated.reason;
    EXPECT_EQ(evaluated.uhd_id, described.uhd_id);
    EXPECT_GT(scoredKernels.load(), 0);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

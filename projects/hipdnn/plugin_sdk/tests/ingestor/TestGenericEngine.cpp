// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

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

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

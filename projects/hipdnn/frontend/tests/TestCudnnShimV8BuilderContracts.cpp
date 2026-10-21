// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Contracts of the v0.x builders over the shim core and the mock hipDNN backend: the error
// boundary, what get_raw_desc() returns per class, plan lifetime and descriptor ownership,
// engine notes through the filters, and hipDNN's last error never leaking into a later
// builder message.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include "fake_backend/ShimV8BuilderPlanFixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;
namespace v8 = fe::detail::v8;

using hipdnn_shim_test::BuilderFailure;
using hipdnn_shim_test::builderFailureOf;
using ::testing::_;

constexpr std::int64_t KNOB_ENGINE = 1;

// The status of the cudnnException `action` threw, SUCCESS when it threw nothing; any
// other exception type escaping a builder is a test failure.
template <typename Action>
cudnnStatus_t thrownStatus(Action&& action)
{
    try
    {
        std::forward<Action>(action)();
    }
    catch(fe::cudnnException& exception)
    {
        return exception.getCudnnStatus();
    }
    catch(const std::exception& exception)
    {
        ADD_FAILURE() << "a non-cudnnException escaped: " << exception.what();
    }
    catch(...)
    {
        ADD_FAILURE() << "a non-exception type escaped";
    }
    return CUDNN_STATUS_SUCCESS;
}

void choose(fe::Engine_v8& built, cudnnBackendKnobType_t type, std::int64_t choice)
{
    auto& knobs = built.getSupportedKnobs();
    const auto knob = std::find_if(knobs.begin(), knobs.end(), [type](const auto& candidate) {
        return candidate.getKnobType() == type;
    });
    ASSERT_NE(knob, knobs.end());
    knob->setChoice(static_cast<std::uint64_t>(choice));
}

class TestCudnnShimV8BuilderContracts : public hipdnn_shim_test::ShimV8BuilderPlanFixture
{
protected:
    // Creating a hipDNN descriptor of `type` raises whatever `raise` throws.
    void createThrows(hipdnnBackendDescriptorType_t type, std::function<void()> raise)
    {
        ON_CALL(*_mockBackend, backendCreateDescriptor(type, _))
            .WillByDefault(
                [raise = std::move(raise)](hipdnnBackendDescriptorType_t,
                                           hipdnnBackendDescriptor_t*) -> hipdnnStatus_t {
                    raise();
                    return HIPDNN_STATUS_INTERNAL_ERROR;
                });
    }

    void createSucceeds(hipdnnBackendDescriptorType_t type)
    {
        ON_CALL(*_mockBackend, backendCreateDescriptor(type, _))
            .WillByDefault(
                [this](hipdnnBackendDescriptorType_t created, hipdnnBackendDescriptor_t* out) {
                    *out = newMockDescriptor(created);
                    return HIPDNN_STATUS_SUCCESS;
                });
    }

    // Records every hipDNN descriptor destroyed, in order.
    void recordDestroys()
    {
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault([this](hipdnnBackendDescriptor_t descriptor) {
                _destroyed.push_back(descriptor);
                return HIPDNN_STATUS_SUCCESS;
            });
    }

    std::size_t destroyCount(hipdnnBackendDescriptor_t descriptor) const
    {
        return static_cast<std::size_t>(
            std::count(_destroyed.begin(), _destroyed.end(), descriptor));
    }

    void expectNoDoubleDestroy() const
    {
        std::map<hipdnnBackendDescriptor_t, int> counts;
        for(hipdnnBackendDescriptor_t descriptor : _destroyed)
        {
            EXPECT_EQ(++counts[descriptor], 1) << "hipDNN descriptor destroyed twice";
        }
    }

    bool isFinalized(hipdnnBackendDescriptor_t descriptor) const
    {
        const std::vector<hipdnnBackendDescriptor_t> all = finalized();
        return std::find(all.begin(), all.end(), descriptor) != all.end();
    }

    void installKnobEngine()
    {
        installEngineKnobs(KNOB_ENGINE, {{"prov.tile_size", 8, 0, 64, 8, {}}});
    }

    static fe::ConvDesc_v8 convDesc()
    {
        const std::array<std::int64_t, 2> pads = {1, 1};
        const std::array<std::int64_t, 2> ones = {1, 1};
        return fe::ConvDescBuilder_v8()
            .setComputeType(fe::DataType_t::FLOAT)
            .setMathMode(CUDNN_CROSS_CORRELATION)
            .setSpatialDimCount(2)
            .setPrePadding(2, pads.data())
            .setPostPadding(2, pads.data())
            .setDilation(2, ones.data())
            .setSpatialStride(2, ones.data())
            .build();
    }

    static fe::ResampleDesc_v8 resampleDesc()
    {
        const std::array<std::int64_t, 2> window = {2, 2};
        const std::array<std::int64_t, 2> zeros = {0, 0};
        return fe::ResampleDescBuilder_v8()
            .setResampleMode(fe::ResampleMode_t::MAXPOOL)
            .setComputeType(fe::DataType_t::FLOAT)
            .setPaddingMode(CUDNN_NEG_INF_PAD)
            .setSpatialDim(2, window.data())
            .setPrePadding(2, zeros.data())
            .setPostPadding(2, zeros.data())
            .setSpatialStride(2, window.data())
            .build();
    }

    std::vector<hipdnnBackendDescriptor_t> _destroyed;
};

// ---- error boundary: every builder and lazy getter throws cudnnException only ----

TEST_F(TestCudnnShimV8BuilderContracts, DescriptorBuildersReportCoreErrors)
{
    EXPECT_EQ(thrownStatus([] {
                  const std::array<std::int64_t, 4> dims = {1, 2, 3, 4};
                  const std::array<std::int64_t, 4> strides = {24, 12, 4, 1};
                  fe::TensorBuilder_v8()
                      .setDataType(fe::DataType_t::FLOAT)
                      .setDim(4, dims.data())
                      .setStride(4, strides.data())
                      .setId(1)
                      .build();
              }),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(thrownStatus([] {
                  fe::ConvDescBuilder_v8()
                      .setComputeType(fe::DataType_t::FLOAT)
                      .setMathMode(CUDNN_CROSS_CORRELATION)
                      .setSpatialDimCount(0)
                      .build();
              }),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(thrownStatus([] {
                  fe::PointWiseDescBuilder_v8()
                      .setMode(fe::PointwiseMode_t::RELU_FWD)
                      .setComputeType(fe::DataType_t::FLOAT)
                      .setReluLowerClipSlope(std::numeric_limits<double>::quiet_NaN())
                      .build();
              }),
              CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(thrownStatus([] {
                  fe::MatMulDescBuilder_v8()
                      .setComputeType(fe::DataType_t::FLOAT)
                      .setPaddingValue(1.5)
                      .build();
              }),
              CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(thrownStatus([] {
                  fe::ReductionDescBuilder_v8().setComputeType(fe::DataType_t::FLOAT).build();
              }),
              CUDNN_STATUS_INVALID_VALUE);
    EXPECT_EQ(thrownStatus([] {
                  const std::array<cudnnFraction_t, 2> window = {{{3, 2}, {3, 2}}};
                  const std::array<cudnnFraction_t, 2> zeros = {{{0, 1}, {0, 1}}};
                  fe::ResampleDescBuilder_v8()
                      .setResampleMode(CUDNN_RESAMPLE_MAXPOOL)
                      .setComputeType(CUDNN_DATA_FLOAT)
                      .setPaddingMode(fe::PaddingMode_t::NEG_INF_PAD)
                      .setSpatialDim(2, window.data())
                      .setPrePadding(2, zeros.data())
                      .setPostPadding(2, zeros.data())
                      .setSpatialStride(2, window.data())
                      .build();
              }),
              CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        thrownStatus([] { fe::RngDescBuilder_v8().setBernoulliDistProbability(0.5).build(); }),
        CUDNN_STATUS_INVALID_VALUE);
    EXPECT_EQ(thrownStatus([] {
                  fe::OperationBuilder_v8(
                      fe::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                      .setxDesc(tensor(X_UID, {2, 4, 8, 8}))
                      .setwDesc(tensor(W_UID, {8, 4, 3, 3}))
                      .setcDesc(convDesc())
                      .build();
              }),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderContracts, GraphBuilderMapsAnyCoreException)
{
    const fe::Operation_v8 op = convForward({2, 4, 8, 8}, {8, 4, 3, 3}, {2, 8, 8, 8});

    createThrows(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, [] { throw std::bad_alloc(); });
    EXPECT_EQ(thrownStatus([&] { graphOf(op); }), CUDNN_STATUS_ALLOC_FAILED);

    createThrows(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                 [] { throw std::runtime_error("unexpected"); });
    EXPECT_EQ(thrownStatus([&] { graphOf(op); }), CUDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestCudnnShimV8BuilderContracts, EngineLayerBuildersMapAnyCoreException)
{
    {
        fe::OperationGraph_v8 graph = convGraph();
        createThrows(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, [] { throw std::bad_alloc(); });
        EXPECT_EQ(thrownStatus([&] {
                      fe::EngineHeuristicsBuilder_v8()
                          .setOperationGraph(graph)
                          .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                          .build();
                  }),
                  CUDNN_STATUS_ALLOC_FAILED);
        createThrows(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                     [] { throw std::runtime_error("unexpected"); });
        EXPECT_EQ(thrownStatus([&] {
                      fe::EngineFallbackListBuilder_v8()
                          .setOperationGraph(graph)
                          .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                          .build();
                  }),
                  CUDNN_STATUS_INTERNAL_ERROR);
        // The lazy getter enumerates too (the failed enumerations above are retried).
        EXPECT_EQ(thrownStatus([&] { graph.getEngineCount(); }), CUDNN_STATUS_INTERNAL_ERROR);
    }
    createSucceeds(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);

    installKnobEngine();
    const fe::OperationGraph_v8 graph = convGraph();
    EXPECT_EQ(thrownStatus([&] { engine(graph, 99); }), CUDNN_STATUS_BAD_PARAM);

    fe::Engine_v8 built = engine(graph, KNOB_ENGINE);
    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 16);
    createThrows(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, [] { throw std::bad_alloc(); });
    EXPECT_EQ(thrownStatus([&] { fe::EngineConfigBuilder_v8().setEngine(built).build(); }),
              CUDNN_STATUS_ALLOC_FAILED);
}

TEST_F(TestCudnnShimV8BuilderContracts, PlanLayerBuildersMapAnyCoreException)
{
    installSerializedPlan({1, 2, 3});
    fe::OperationGraph_v8 graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 3U);
    const fe::ExecutionPlan_v8 plan = planFor(configs[0], graph.getTag());

    ON_CALL(*_mockBackend, backendGetSerializedExecutionPlanExt(_, _, _, _))
        .WillByDefault([](hipdnnBackendDescriptor_t, std::size_t, std::size_t*, std::uint8_t*)
                           -> hipdnnStatus_t { throw std::runtime_error("unexpected"); });
    EXPECT_EQ(thrownStatus([&] { plan.getJsonRepresentation(); }), CUDNN_STATUS_INTERNAL_ERROR);

    EXPECT_EQ(
        thrownStatus([&] { fe::ExecutionPlanBuilder_v8().setHandle(_handle).loadFromJson("{"); }),
        CUDNN_STATUS_BAD_PARAM);

    createThrows(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                 [] { throw std::runtime_error("unexpected"); });
    EXPECT_EQ(thrownStatus([&] { planFor(configs[1], graph.getTag()); }),
              CUDNN_STATUS_INTERNAL_ERROR);

    createThrows(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, [] { throw std::bad_alloc(); });
    EXPECT_EQ(thrownStatus([&] { variantPack(); }), CUDNN_STATUS_ALLOC_FAILED);
}

// ---- get_raw_desc() per class ----

TEST_F(TestCudnnShimV8BuilderContracts, RawDescriptorFollowsClassTable)
{
    installKnobEngine();
    installSerializedPlan({1, 2, 3});

    const fe::Tensor_v8 x = tensor(X_UID, {2, 4, 8, 8});
    EXPECT_EQ(x.get_raw_desc(), nullptr);
    EXPECT_NE(v8::recordOf<v8::TensorRecord>(x.get_desc()), nullptr);
    const fe::ConvDesc_v8 conv = convDesc();
    EXPECT_EQ(conv.get_raw_desc(), nullptr);
    EXPECT_NE(v8::recordOf<v8::ConvolutionRecord>(conv.get_desc()), nullptr);
    const fe::PointWiseDesc_v8 relu = fe::PointWiseDescBuilder_v8()
                                          .setMode(fe::PointwiseMode_t::RELU_FWD)
                                          .setComputeType(fe::DataType_t::FLOAT)
                                          .build();
    EXPECT_EQ(relu.get_raw_desc(), nullptr);
    EXPECT_NE(v8::recordOf<v8::PointwiseRecord>(relu.get_desc()), nullptr);
    const fe::MatMulDesc_v8 matmul
        = fe::MatMulDescBuilder_v8().setComputeType(fe::DataType_t::FLOAT).build();
    EXPECT_EQ(matmul.get_raw_desc(), nullptr);
    EXPECT_NE(v8::recordOf<v8::MatmulRecord>(matmul.get_desc()), nullptr);
    const fe::ReductionDesc_v8 reduction = fe::ReductionDescBuilder_v8()
                                               .setComputeType(fe::DataType_t::FLOAT)
                                               .setReductionOp(fe::ReductionMode_t::ADD)
                                               .build();
    EXPECT_EQ(reduction.get_raw_desc(), nullptr);
    EXPECT_NE(v8::recordOf<v8::ReductionRecord>(reduction.get_desc()), nullptr);
    const fe::ResampleDesc_v8 resample = resampleDesc();
    EXPECT_EQ(resample.get_raw_desc(), nullptr);
    EXPECT_NE(v8::recordOf<v8::ResampleRecord>(resample.get_desc()), nullptr);
    const fe::RngDesc_v8 rng = fe::RngDescBuilder_v8()
                                   .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                   .setBernoulliDistProbability(0.25)
                                   .build();
    EXPECT_EQ(rng.get_raw_desc(), nullptr);
    EXPECT_NE(v8::recordOf<v8::RngRecord>(rng.get_desc()), nullptr);
    const fe::Operation_v8 op = convForward({2, 4, 8, 8}, {8, 4, 3, 3}, {2, 8, 8, 8});
    EXPECT_EQ(op.get_raw_desc(), nullptr);
    EXPECT_NE(v8::recordOf<v8::ConvolutionForwardOperationRecord>(op.get_desc()), nullptr);

    fe::OperationGraph_v8 graph = graphOf(op);
    const auto core = v8::recordOf<v8::GraphCore>(graph.get_desc());
    ASSERT_NE(core, nullptr);
    EXPECT_EQ(graph.get_raw_desc(), core->hipdnnGraph());
    EXPECT_EQ(typeOf(graph.get_raw_desc()), HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    EXPECT_TRUE(isFinalized(graph.get_raw_desc()));

    fe::EngineHeuristics_v8 heuristics = fe::EngineHeuristicsBuilder_v8()
                                             .setOperationGraph(graph)
                                             .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                             .build();
    EXPECT_EQ(heuristics.get_raw_desc(), nullptr);
    const fe::EngineFallbackList_v8 fallback
        = fe::EngineFallbackListBuilder_v8()
              .setOperationGraph(graph)
              .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
              .build();
    EXPECT_EQ(fallback.get_raw_desc(), nullptr);

    const fe::EngineConfigList results = heuristics.getEngineConfig(3);
    ASSERT_EQ(results.size(), 3U);
    for(std::size_t i = 0; i < results.size(); ++i)
    {
        const auto id = static_cast<std::int64_t>(i);
        hipdnnBackendDescriptor_t raw = results[i]->get_backend_descriptor();
        EXPECT_EQ(typeOf(raw), HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        EXPECT_EQ(engineIdOf(raw), id);
        EXPECT_EQ(raw, core->defaultEngineConfig(id));
        EXPECT_NE(v8::recordOf<v8::ConfigRecord>(results[i]), nullptr);
    }

    fe::Engine_v8 built = engine(graph, KNOB_ENGINE);
    EXPECT_EQ(typeOf(built.get_raw_desc()), HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
    EXPECT_EQ(built.get_raw_desc(), core->defaultEngine(KNOB_ENGINE));
    EXPECT_EQ(engineIdOf(built.get_raw_desc()), KNOB_ENGINE);
    EXPECT_NE(v8::recordOf<v8::EngineRecord>(built.get_desc()), nullptr);

    const fe::EngineConfig_v8 knobFree = fe::EngineConfigBuilder_v8().setEngine(built).build();
    EXPECT_EQ(knobFree.get_raw_desc(), results[KNOB_ENGINE]->get_backend_descriptor());
    EXPECT_NE(v8::recordOf<v8::ConfigRecord>(knobFree.get_desc()), nullptr);

    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 16);
    const fe::EngineConfig_v8 knobbed = fe::EngineConfigBuilder_v8().setEngine(built).build();
    hipdnnBackendDescriptor_t knobbedRaw = knobbed.get_raw_desc();
    EXPECT_EQ(typeOf(knobbedRaw), HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    EXPECT_NE(knobbedRaw, knobFree.get_raw_desc());
    EXPECT_EQ(engineIdOf(knobbedRaw), KNOB_ENGINE);
    // Built and finalized by EngineConfigBuilder::build(), not at plan build.
    EXPECT_TRUE(isFinalized(knobbedRaw));
    EXPECT_NE(v8::recordOf<v8::ConfigRecord>(knobbed.get_desc()), nullptr);

    const fe::ExecutionPlan_v8 plan
        = fe::ExecutionPlanBuilder_v8().setEngineConfig(knobbed).build();
    const auto planRecord = v8::recordOf<v8::PlanRecord>(plan.get_desc());
    ASSERT_NE(planRecord, nullptr);
    EXPECT_EQ(plan.get_raw_desc(), planRecord->hipdnnPlan());
    EXPECT_EQ(typeOf(plan.get_raw_desc()), HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    EXPECT_TRUE(isFinalized(plan.get_raw_desc()));

    const fe::ExecutionPlan_v8 loaded
        = fe::ExecutionPlanBuilder_v8().setHandle(_handle).loadFromJson(
            plan.getJsonRepresentation());
    EXPECT_EQ(typeOf(loaded.get_raw_desc()), HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    EXPECT_NE(loaded.get_raw_desc(), plan.get_raw_desc());
    EXPECT_NE(v8::recordOf<v8::PlanRecord>(loaded.get_desc()), nullptr);

    const fe::VariantPack_v8 pack = variantPack();
    EXPECT_EQ(typeOf(pack.get_raw_desc()), HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    EXPECT_TRUE(isFinalized(pack.get_raw_desc()));
    EXPECT_NE(v8::recordOf<v8::VariantPackRecord>(pack.get_desc()), nullptr);
}

// ---- lifetime and ownership ----

TEST_F(TestCudnnShimV8BuilderContracts, PlanOutlivesEverythingItWasBuiltFrom)
{
    recordDestroys();
    std::vector<fe::ExecutionPlan_v8> kept;
    hipdnnBackendDescriptor_t hipdnnGraph = nullptr;
    {
        fe::OperationGraph_v8 graph = convGraph();
        hipdnnGraph = graph.get_raw_desc();
        fe::EngineHeuristics_v8 heuristics = fe::EngineHeuristicsBuilder_v8()
                                                 .setOperationGraph(graph)
                                                 .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                                 .build();
        const fe::EngineConfigList configs = heuristics.getEngineConfig(3);
        ASSERT_EQ(configs.size(), 3U);
        fe::ExecutionPlanBuilder_v8 builder;
        builder.setEngineConfig(configs[2], graph.getTag());
        const fe::ExecutionPlan_v8 original = builder.build();
        kept.push_back(original);
        kept.push_back(original);
    }
    EXPECT_EQ(destroyCount(hipdnnGraph), 0U);
    // The copy outlives both the original and the other copy.
    kept.erase(kept.begin());

    hipdnnBackendDescriptor_t hipdnnPlan = kept.front().get_raw_desc();
    const fe::VariantPack_v8 pack = variantPack();
    EXPECT_EQ(cudnnBackendExecute(_handle, hipdnnPlan, pack.get_raw_desc()), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(executedEngineIds(), std::vector<std::int64_t>{2});
    EXPECT_EQ(destroyCount(hipdnnPlan), 0U);

    kept.clear();
    EXPECT_EQ(destroyCount(hipdnnPlan), 1U);
    EXPECT_EQ(destroyCount(hipdnnGraph), 1U);
    expectNoDoubleDestroy();
}

TEST_F(TestCudnnShimV8BuilderContracts, SharedDefaultConfigDestroyedOnce)
{
    recordDestroys();
    hipdnnBackendDescriptor_t shared = nullptr;
    {
        fe::OperationGraph_v8 graph = convGraph();
        const fe::EngineConfigList first = engineConfigs(graph);
        const fe::EngineConfigList second = engineConfigs(graph);
        ASSERT_EQ(first.size(), 3U);
        ASSERT_EQ(second.size(), 3U);
        shared = first[1]->get_backend_descriptor();
        EXPECT_EQ(second[1]->get_backend_descriptor(), shared);
        const fe::EngineConfig_v8 viaEngine
            = fe::EngineConfigBuilder_v8().setEngine(engine(graph, 1)).build();
        EXPECT_EQ(viaEngine.get_raw_desc(), shared);
        const fe::ExecutionPlan_v8 a = planFor(first[1], graph.getTag());
        const fe::ExecutionPlan_v8 b = planFor(second[1], graph.getTag());
        EXPECT_NE(a.get_raw_desc(), b.get_raw_desc());
    }
    EXPECT_EQ(destroyCount(shared), 1U);
    expectNoDoubleDestroy();
}

// ---- engine notes through the filters ----

TEST_F(TestCudnnShimV8BuilderContracts, DeterministicFilterKeepsOnlyClaimedEngines)
{
    const std::int64_t deterministic = hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID;
    installHeuristicResults({deterministic + 3, deterministic, -4, deterministic + 1, 0});
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineConfigList candidates = engineConfigs(graph);
    ASSERT_EQ(candidates.size(), 5U);
    fe::EngineConfigList kept;

    // filter() moves the configs the predicate rejects into `kept`.
    fe::filter(candidates, kept, [](cudnnBackendDescriptor_t config) {
        return fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(config);
    });

    ASSERT_EQ(kept.size(), 1U);
    EXPECT_EQ(engineIdOf(kept[0]->get_backend_descriptor()), deterministic);
    ASSERT_EQ(candidates.size(), 4U);
    for(const fe::ManagedOpaqueDescriptor& config : candidates)
    {
        EXPECT_NE(engineIdOf(config->get_backend_descriptor()), deterministic);
    }
}

TEST_F(TestCudnnShimV8BuilderContracts, NotesReadOnKnobConfig)
{
    installKnobEngine();
    installBehaviorNotes(KNOB_ENGINE, {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    const fe::OperationGraph_v8 graph = convGraph();
    fe::Engine_v8 built = engine(graph, KNOB_ENGINE);
    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 16);
    const fe::EngineConfig_v8 knobbed = fe::EngineConfigBuilder_v8().setEngine(built).build();
    hipdnnBackendDescriptor_t raw = knobbed.get_raw_desc();
    ASSERT_NE(raw, nullptr);

    EXPECT_TRUE(fe::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION>(raw));
    EXPECT_FALSE(fe::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER>(raw));
    EXPECT_TRUE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(raw));
    // The numerical note is decided by engine id; hipDNN's NUMERICAL_NOTE is never read.
    const std::vector<hipdnnBackendAttributeName_t> gets = getAttributeNames();
    EXPECT_EQ(std::count(gets.begin(), gets.end(), HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE), 0);
}

// ---- hipDNN's last error never reaches a later builder message ----

TEST_F(TestCudnnShimV8BuilderContracts, DroppedEngineLeavesNoHipdnnText)
{
    installHeuristicResults({0, 1, 2}, {1});
    const fe::OperationGraph_v8 graph = convGraph();
    ASSERT_EQ(graph.getEngineCount(), 2);

    const BuilderFailure failure = builderFailureOf([] {
        const std::array<std::int64_t, 4> dims = {1, 2, 3, 4};
        fe::TensorBuilder_v8().setDataType(fe::DataType_t::FLOAT).setDim(4, dims.data()).build();
    });
    ASSERT_TRUE(failure.thrown);
    // The mock's text for the dropped engine is "mock engine config 1 failed to finalize".
    EXPECT_EQ(failure.what.find("mock engine config"), std::string::npos) << failure.what;
}

TEST_F(TestCudnnShimV8BuilderContracts, FailedFilterGetLeavesNoHipdnnText)
{
    std::string pending;
    ON_CALL(*_mockBackend, getLastErrorString(_, _))
        .WillByDefault([&pending](char* message, std::size_t size) {
            const std::size_t length = std::min(pending.size(), size - 1);
            std::copy_n(pending.begin(), length, message);
            message[length] = '\0';
            pending.clear();
        });
    hipdnnBackendDescriptor_t foreign = newMockDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    ON_CALL(*_mockBackend, backendGetAttribute(foreign, _, _, _, _, _))
        .WillByDefault([&pending](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  std::int64_t,
                                  std::int64_t*,
                                  void*) {
            pending = "hipDNN rejected the GET";
            return HIPDNN_STATUS_BAD_PARAM;
        });

    EXPECT_FALSE(fe::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(foreign));
    EXPECT_FALSE(fe::hasBehaviorNote<CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION>(foreign));

    const BuilderFailure failure = builderFailureOf(
        [] { fe::ConvDescBuilder_v8().setComputeType(fe::DataType_t::FLOAT).build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.what.find("hipDNN rejected"), std::string::npos) << failure.what;
}

} // namespace

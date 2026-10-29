// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The builders' error boundary with exceptions disabled: a core error, or any exception a
// hipDNN call raises inside a build or a lazy getter, becomes the object's status.
#ifndef NV_CUDNN_DISABLE_EXCEPTION
#error "TestCudnnShimV8NoexceptBoundary.cpp must be built with NV_CUDNN_DISABLE_EXCEPTION"
#endif

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fake_backend/ShimV8BuilderPlanFixture.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using ::testing::_;

class TestCudnnShimV8NoexceptBoundary : public hipdnn_shim_test::ShimV8BuilderPlanFixture
{
};

TEST_F(TestCudnnShimV8NoexceptBoundary, BuildersReturnCoreStatus)
{
    const std::array<std::int64_t, 4> dims = {1, 2, 3, 4};
    const std::array<std::int64_t, 4> strides = {24, 12, 4, 1};
    const fe::Tensor tensor = fe::TensorBuilder()
                                  .setDataType(fe::DataType_t::FLOAT)
                                  .setDim(4, dims.data())
                                  .setStride(4, strides.data())
                                  .setId(1)
                                  .setAlignment(16)
                                  .setVirtual()
                                  .setByValue()
                                  .build();
    EXPECT_EQ(tensor.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(tensor.get_raw_desc(), nullptr);

    const fe::MatMulDesc matmul = fe::MatMulDescBuilder()
                                      .setComputeType(fe::DataType_t::FLOAT)
                                      .setPaddingValue(1.5)
                                      .build();
    EXPECT_EQ(matmul.get_status(), CUDNN_STATUS_NOT_SUPPORTED);

    const fe::Operation op = convForward({2, 4, 8, 8}, {8, 4, 3, 3}, {2, 8, 8, 8});
    ASSERT_EQ(op.get_status(), CUDNN_STATUS_SUCCESS) << op.get_error();

    createThrows(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, [] { throw std::bad_alloc(); });
    const fe::OperationGraph graph = graphOf(op);
    EXPECT_EQ(graph.get_status(), CUDNN_STATUS_ALLOC_FAILED);
    // A failed graph has no engine space.
    EXPECT_EQ(graph.getEngineCount(), -1);
    EXPECT_EQ(graph.get_status(), CUDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestCudnnShimV8NoexceptBoundary, DescriptorAndOperationBuildersReturnCoreStatus)
{
    const fe::ConvDesc conv = fe::ConvDescBuilder()
                                  .setComputeType(fe::DataType_t::FLOAT)
                                  .setMathMode(CUDNN_CROSS_CORRELATION)
                                  .setSpatialDimCount(0)
                                  .build();
    EXPECT_EQ(conv.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(conv.get_raw_desc(), nullptr);

    const fe::PointWiseDesc relu
        = fe::PointWiseDescBuilder()
              .setMode(fe::PointwiseMode_t::RELU_FWD)
              .setComputeType(fe::DataType_t::FLOAT)
              .setReluLowerClipSlope(std::numeric_limits<double>::quiet_NaN())
              .build();
    EXPECT_EQ(relu.get_status(), CUDNN_STATUS_BAD_PARAM);

    const fe::ReductionDesc reduction
        = fe::ReductionDescBuilder().setComputeType(fe::DataType_t::FLOAT).build();
    EXPECT_EQ(reduction.get_status(), CUDNN_STATUS_INVALID_VALUE);

    const std::array<cudnnFraction_t, 2> window = {{{3, 2}, {3, 2}}};
    const std::array<cudnnFraction_t, 2> zeros = {{{0, 1}, {0, 1}}};
    const fe::ResampleDesc resample = fe::ResampleDescBuilder()
                                          .setResampleMode(CUDNN_RESAMPLE_MAXPOOL)
                                          .setComputeType(CUDNN_DATA_FLOAT)
                                          .setPaddingMode(fe::PaddingMode_t::NEG_INF_PAD)
                                          .setSpatialDim(2, window.data())
                                          .setPrePadding(2, zeros.data())
                                          .setPostPadding(2, zeros.data())
                                          .setSpatialStride(2, window.data())
                                          .build();
    EXPECT_EQ(resample.get_status(), CUDNN_STATUS_NOT_SUPPORTED);

    const fe::RngDesc rng = fe::RngDescBuilder().setBernoulliDistProbability(0.5).build();
    EXPECT_EQ(rng.get_status(), CUDNN_STATUS_INVALID_VALUE);

    const fe::Operation op
        = fe::OperationBuilder(fe::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
              .setxDesc(tensor(X_UID, {2, 4, 8, 8}))
              .setwDesc(tensor(W_UID, {8, 4, 3, 3}))
              .build();
    EXPECT_EQ(op.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(op.get_raw_desc(), nullptr);
}

TEST_F(TestCudnnShimV8NoexceptBoundary, EngineLayerReturnsStatus)
{
    fe::OperationGraph graph = convGraph();
    ASSERT_EQ(graph.get_status(), CUDNN_STATUS_SUCCESS) << graph.get_error();

    createThrows(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                 [] { throw std::runtime_error("unexpected"); });
    fe::EngineHeuristics heuristics = fe::EngineHeuristicsBuilder()
                                          .setOperationGraph(graph)
                                          .setHeurMode(CUDNN_HEUR_MODE_FALLBACK)
                                          .build();
    EXPECT_EQ(heuristics.get_status(), CUDNN_STATUS_INTERNAL_ERROR);
    EXPECT_TRUE(heuristics.getEngineConfig(4).empty());
    EXPECT_EQ(heuristics.get_status(), CUDNN_STATUS_NOT_INITIALIZED);

    graph.getEngineCount();
    EXPECT_EQ(graph.get_status(), CUDNN_STATUS_INTERNAL_ERROR);

    const fe::Engine engine
        = fe::EngineBuilder().setOperationGraph(graph).setGlobalEngineIdx(-1).build();
    EXPECT_EQ(engine.get_status(), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8NoexceptBoundary, FallbackListAndKnobConfigReturnStatus)
{
    {
        fe::OperationGraph graph = convGraph();
        createThrows(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                     [] { throw std::runtime_error("unexpected"); });
        fe::EngineFallbackList fallback
            = fe::EngineFallbackListBuilder()
                  .setOperationGraph(graph)
                  .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                  .build();
        // As upstream, the inner heuristics' status is not copied onto the list without
        // exceptions: the core error stays inside, and the list is empty.
        EXPECT_EQ(fallback.get_status(), CUDNN_STATUS_SUCCESS);
        EXPECT_TRUE(fallback.getFallbackList().empty());
    }
    ON_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillByDefault([this](hipdnnBackendDescriptorType_t type, hipdnnBackendDescriptor_t* out) {
            *out = newMockDescriptor(type);
            return HIPDNN_STATUS_SUCCESS;
        });

    installEngineKnobs(1, {{"prov.tile_size", 8, 0, 64, 8, {}}});
    const fe::OperationGraph graph = convGraph();
    fe::Engine built = engine(graph, 1);
    ASSERT_EQ(built.get_status(), CUDNN_STATUS_SUCCESS) << built.get_error();
    auto& knobs = built.getSupportedKnobs();
    ASSERT_EQ(knobs.size(), 1U);
    knobs.front().setChoice(16);
    createThrows(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, [] { throw std::bad_alloc(); });
    const fe::EngineConfig config = fe::EngineConfigBuilder().setEngine(built).build();
    EXPECT_EQ(config.get_status(), CUDNN_STATUS_ALLOC_FAILED);
    EXPECT_EQ(config.get_raw_desc(), nullptr);
}

TEST_F(TestCudnnShimV8NoexceptBoundary, PlanLayerReturnsStatus)
{
    installSerializedPlan({1, 2, 3});
    fe::OperationGraph graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 3U);
    const std::string planJson = planFor(configs[0], graph.getTag()).getJsonRepresentation();
    ASSERT_FALSE(planJson.empty());

    createThrows(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, [] { throw std::bad_alloc(); });
    const fe::ExecutionPlan failed = planFor(configs[0], graph.getTag());
    EXPECT_EQ(failed.get_status(), CUDNN_STATUS_ALLOC_FAILED);
    EXPECT_EQ(failed.get_raw_desc(), nullptr);
    EXPECT_TRUE(failed.getJsonRepresentation().empty());
    EXPECT_EQ(failed.get_status(), CUDNN_STATUS_NOT_INITIALIZED);

    // Rejected by the JSON parser before any hipDNN call.
    const fe::ExecutionPlan unparsed
        = fe::ExecutionPlanBuilder().setHandle(_handle).loadFromJson("{");
    EXPECT_EQ(unparsed.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(unparsed.get_raw_desc(), nullptr);

    setDeserializeStatus(HIPDNN_STATUS_ALLOC_FAILED);
    const fe::ExecutionPlan undeserialized
        = fe::ExecutionPlanBuilder().setHandle(_handle).loadFromJson(planJson);
    EXPECT_EQ(undeserialized.get_status(), CUDNN_STATUS_ALLOC_FAILED);
    EXPECT_NE(std::string(undeserialized.get_error()).find("deserializing"), std::string::npos)
        << undeserialized.get_error();
    EXPECT_EQ(undeserialized.get_raw_desc(), nullptr);
    setDeserializeStatus(HIPDNN_STATUS_SUCCESS);

    // Upstream's own prechecks (no config, no handle) also leave a null raw descriptor.
    const fe::ExecutionPlan noConfig = fe::ExecutionPlanBuilder().build();
    EXPECT_EQ(noConfig.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(noConfig.get_raw_desc(), nullptr);
    const fe::ExecutionPlan noHandle = fe::ExecutionPlanBuilder().loadFromJson("{}");
    EXPECT_EQ(noHandle.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(noHandle.get_raw_desc(), nullptr);

    createThrows(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                 [] { throw std::runtime_error("unexpected"); });
    const fe::VariantPack pack = variantPack();
    EXPECT_EQ(pack.get_status(), CUDNN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(pack.get_raw_desc(), nullptr);
}

// A failed variant pack has no hipDNN descriptor; the passthrough hands hipDNN the null
// pack, which hipDNN rejects.
TEST_F(TestCudnnShimV8NoexceptBoundary, ExecuteWithFailedVariantPackIsBadParam)
{
    fe::OperationGraph graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 3U);
    const fe::ExecutionPlan plan = planFor(configs[0], graph.getTag());
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    const std::array<std::int64_t, 2> uids = {1, 2};
    std::array<void*, 1> pointers = {nullptr};
    const fe::VariantPack pack = fe::VariantPackBuilder()
                                     .setDataPointers(1, pointers.data())
                                     .setUids(2, uids.data())
                                     .build();
    ASSERT_EQ(pack.get_status(), CUDNN_STATUS_BAD_PARAM);
    ASSERT_EQ(pack.get_raw_desc(), nullptr);
    ON_CALL(*_mockBackend, backendExecute(_, _, nullptr))
        .WillByDefault(::testing::Return(HIPDNN_STATUS_BAD_PARAM));

    EXPECT_EQ(cudnnBackendExecute(_handle, plan.get_raw_desc(), pack.get_raw_desc()),
              CUDNN_STATUS_BAD_PARAM);
}

// Upstream hands a negative count to reserve()/resize(), which throws std::length_error even
// with exceptions disabled. The shim records BAD_PARAM on the object instead; build() then
// reports the missing operands with the same status.
TEST_F(TestCudnnShimV8NoexceptBoundary, NegativeCountsSetBadParam)
{
    std::array<void*, 1> pointers = {nullptr};
    const std::array<std::int64_t, 1> uids = {1};

    const fe::VariantPack dataPointers
        = fe::VariantPackBuilder().setDataPointers(-1, pointers.data()).build();
    EXPECT_EQ(dataPointers.get_status(), CUDNN_STATUS_BAD_PARAM);

    const fe::VariantPack uidPack = fe::VariantPackBuilder().setUids(-1, uids.data()).build();
    EXPECT_EQ(uidPack.get_status(), CUDNN_STATUS_BAD_PARAM);

    const fe::OperationGraph graph
        = fe::OperationGraphBuilder().setOperationGraph(-1, nullptr).build();
    EXPECT_EQ(graph.get_status(), CUDNN_STATUS_BAD_PARAM);
}

} // namespace

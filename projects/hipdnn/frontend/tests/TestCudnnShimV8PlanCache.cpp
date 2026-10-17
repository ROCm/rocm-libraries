// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x ExecutionPlanCache over plans built through the shim and the
// mock hipDNN backend.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_ExecutionPlanCache.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

class TestCudnnShimV8PlanCache : public hipdnn_shim_test::ShimV8PlanFixture
{
protected:
    // conv(x, w) -> virtual y -> pointwise `mode` -> z.
    fe::OperationGraph_v8 convPointwiseGraph(fe::PointwiseMode_t mode)
    {
        const Dims yDims = {2, 8, 8, 8};
        const Dims yStrides = packedStrides(yDims);
        const fe::Tensor_v8 y = fe::TensorBuilder_v8()
                                    .setDataType(fe::DataType_t::FLOAT)
                                    .setDim(4, yDims.data())
                                    .setStride(4, yStrides.data())
                                    .setId(Y_UID)
                                    .setAlignment(16)
                                    .setVirtual(true)
                                    .build();
        const std::array<std::int64_t, 2> pads = {1, 1};
        const std::array<std::int64_t, 2> ones = {1, 1};
        std::vector<fe::Operation_v8> ops;
        ops.push_back(
            fe::OperationBuilder_v8(fe::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(tensor(X_UID, {2, 4, 8, 8}))
                .setwDesc(tensor(W_UID, {8, 4, 3, 3}))
                .setyDesc(y)
                .setcDesc(fe::ConvDescBuilder_v8()
                              .setComputeType(fe::DataType_t::FLOAT)
                              .setMathMode(CUDNN_CROSS_CORRELATION)
                              .setSpatialDimCount(2)
                              .setPrePadding(2, pads.data())
                              .setPostPadding(2, pads.data())
                              .setDilation(2, ones.data())
                              .setSpatialStride(2, ones.data())
                              .build())
                .build());
        ops.push_back(fe::OperationBuilder_v8(fe::DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                          .setpwDesc(fe::PointWiseDescBuilder_v8()
                                         .setMode(mode)
                                         .setComputeType(fe::DataType_t::FLOAT)
                                         .build())
                          .setxDesc(y)
                          .setyDesc(tensor(Y_UID + 1, yDims))
                          .build());
        return fe::OperationGraphBuilder_v8().setHandle(_handle).setOperationGraph(ops).build();
    }
};

TEST_F(TestCudnnShimV8PlanCache, KeyIsFirstFeatureVector)
{
    const fe::OperationGraph_v8 graph = convGraph();
    const fe::OperationGraph_v8 sameShape = convGraph();
    const fe::OperationGraph_v8 otherShape = convGraph(8);
    ASSERT_EQ(sameShape.getFeatureVector(), graph.getFeatureVector());
    ASSERT_NE(otherShape.getFeatureVector(), graph.getFeatureVector());

    fe::ExecutionPlanCache cache("conv");
    EXPECT_EQ(cache.get_name(), "conv");
    const fe::ExecutionPlan_v8 plan = planFor(graph, 1);
    cache.add_plan_to_cache(graph, plan);

    const fe::ExecutionPlan_v8* cached = nullptr;
    ASSERT_TRUE(cache.get_plan_from_cache(sameShape, cached));
    ASSERT_NE(cached, nullptr);
    EXPECT_EQ(cached->get_raw_desc(), plan.get_raw_desc());
    EXPECT_EQ(cached->getTag(), "ConvFwd_eng1");

    const fe::ExecutionPlan_v8* missing = nullptr;
    EXPECT_FALSE(cache.get_plan_from_cache(otherShape, missing));
    EXPECT_EQ(missing, nullptr);
}

// Only the first operation keys the cache: graphs that differ after it share
// the entry.
TEST_F(TestCudnnShimV8PlanCache, LaterOpsDoNotChangeKey)
{
    const fe::OperationGraph_v8 relu = convPointwiseGraph(fe::PointwiseMode_t::RELU_FWD);
    const fe::OperationGraph_v8 tanh = convPointwiseGraph(fe::PointwiseMode_t::TANH_FWD);
    ASSERT_EQ(relu.get_status(), CUDNN_STATUS_SUCCESS) << relu.get_error();
    ASSERT_EQ(tanh.get_status(), CUDNN_STATUS_SUCCESS) << tanh.get_error();
    ASSERT_EQ(relu.getOpCount(), 2U);
    ASSERT_NE(relu.getTag(), tanh.getTag());
    ASSERT_EQ(relu.getFeatureVector(), tanh.getFeatureVector());

    fe::ExecutionPlanCache cache("fused");
    const fe::ExecutionPlan_v8 plan = planFor(relu, 2);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    cache.add_plan_to_cache(relu, plan);

    const fe::ExecutionPlan_v8* cached = nullptr;
    ASSERT_TRUE(cache.get_plan_from_cache(tanh, cached));
    EXPECT_EQ(cached->get_raw_desc(), plan.get_raw_desc());
    EXPECT_EQ(cached->getTag(), relu.getTag() + "eng2");
}

TEST_F(TestCudnnShimV8PlanCache, InsertDoesNotOverwrite)
{
    const fe::OperationGraph_v8 graph = convGraph();
    fe::ExecutionPlanCache cache("conv");
    cache.add_plan_to_cache(graph, planFor(graph, 0));
    cache.add_plan_to_cache(graph, planFor(graph, 2));

    const fe::ExecutionPlan_v8* cached = nullptr;
    ASSERT_TRUE(cache.get_plan_from_cache(graph, cached));
    EXPECT_EQ(cached->getTag(), "ConvFwd_eng0");
}

TEST_F(TestCudnnShimV8PlanCache, SaturationCountGatesStability)
{
    const fe::OperationGraph_v8 graph = convGraph();

    fe::ExecutionPlanCache always("always");
    EXPECT_TRUE(always.is_fastest_plan_stable(graph, "ConvFwd_eng0"));
    EXPECT_TRUE(always.is_fastest_plan_stable(graph, "ConvFwd_eng0"));

    fe::ExecutionPlanCache saturating("saturating");
    saturating.set_saturation_count(3);
    EXPECT_FALSE(saturating.is_fastest_plan_stable(graph, "ConvFwd_eng0"));
    EXPECT_FALSE(saturating.is_fastest_plan_stable(graph, "ConvFwd_eng0"));
    // Each tag counts separately.
    EXPECT_FALSE(saturating.is_fastest_plan_stable(graph, "ConvFwd_eng1"));
    EXPECT_TRUE(saturating.is_fastest_plan_stable(graph, "ConvFwd_eng0"));

    // Once the graph has a cached plan, nothing is stable any more.
    saturating.add_plan_to_cache(graph, planFor(graph, 0));
    EXPECT_FALSE(saturating.is_fastest_plan_stable(graph, "ConvFwd_eng0"));
    EXPECT_FALSE(saturating.is_fastest_plan_stable(graph, "ConvFwd_eng1"));

    fe::ExecutionPlanCache_v1 base("base");
    EXPECT_TRUE(base.is_fastest_plan_stable(graph, "ConvFwd_eng0"));
}

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Pins the shim's graph-lowering rules through the ported v0.x builders only,
// finalizing against the real hipDNN backend. Graph finalize needs no device,
// so this suite does not skip on device-less hosts.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "CudnnShimV8BuilderGraph.hpp"

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;
using namespace hipdnn_shim_v8_test;

using fe::DataType_t;
using fe::DescriptorType_t;
using fe::PointwiseMode_t;

fe::Tensor indexTensor(std::int64_t uid)
{
    TensorOptions options;
    options.dataType = DataType_t::INT32;
    return tensor(uid, POOLED_DIMS, options);
}

fe::Operation relu(std::int64_t xUid, std::int64_t yUid)
{
    return pointwise(
        PointwiseMode_t::RELU_FWD, tensor(xUid, X_DIMS), nullptr, tensor(yUid, X_DIMS));
}

cudnnStatus_t buildStatus(fe::OperationGraphBuilder& builder)
{
    try
    {
        return builder.build().get_status();
    }
    catch(fe::cudnnException& e)
    {
        return e.getCudnnStatus();
    }
}

class IntegrationCudnnShimV8BuilderLowering : public DevicelessGraphFixture
{
protected:
    void expectGraphRejected(const std::vector<fe::Operation>& ops, cudnnStatus_t status)
    {
        fe::OperationGraphBuilder builder;
        builder.setHandle(_handle).setOperationGraph(ops);
        EXPECT_EQ(buildStatus(builder), status);
    }
};

TEST_F(IntegrationCudnnShimV8BuilderLowering, ConvolutionBackwardData)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(
        fe::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
            .setcDesc(convDesc())
            .setdyDesc(tensor(1, Y_DIMS))
            .setwDesc(tensor(2, W_DIMS))
            .setdxDesc(tensor(3, X_DIMS))
            .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, ConvolutionBackwardFilter)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(
        fe::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
            .setcDesc(convDesc())
            .setxDesc(tensor(1, X_DIMS))
            .setdyDesc(tensor(2, Y_DIMS))
            .setdwDesc(tensor(3, W_DIMS))
            .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, PointwiseActivationBackward)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                         .setpwDesc(pointwiseDesc(PointwiseMode_t::RELU_BWD))
                         .setdyDesc(tensor(1, X_DIMS))
                         .setxDesc(tensor(2, X_DIMS))
                         .setdxDesc(tensor(3, X_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, ResampleForwardMaxWithIndex)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                         .setResampleDesc(resampleDesc(CUDNN_RESAMPLE_MAXPOOL))
                         .setxDesc(tensor(1, X_DIMS))
                         .setyDesc(tensor(2, POOLED_DIMS))
                         .setidxDesc(indexTensor(3))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, ResampleBackwardAverage)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                         .setResampleDesc(resampleDesc(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING))
                         .setdyDesc(tensor(1, POOLED_DIMS))
                         .setdxDesc(tensor(2, X_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, ResampleBackwardMaxWithIndex)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                         .setResampleDesc(resampleDesc(CUDNN_RESAMPLE_MAXPOOL))
                         .setdyDesc(tensor(1, POOLED_DIMS))
                         .setidxDesc(indexTensor(2))
                         .setdxDesc(tensor(3, X_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, BatchNormInference)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                         .setNormalizationMode(fe::NormMode_t::BATCH_NORM)
                         .setNormFwdPhase(fe::NormFwdPhase_t::INFERENCE)
                         .setxDesc(tensor(1, X_DIMS))
                         .setSavedMeanAndInvVar(tensor(2, CHANNEL_DIMS), tensor(3, CHANNEL_DIMS))
                         .setScaleAndBias(tensor(4, CHANNEL_DIMS), tensor(5, CHANNEL_DIMS))
                         .setEpsilonTensor(scalar(6))
                         .setyDesc(tensor(7, X_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, BatchNormBackward)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                         .setNormalizationMode(fe::NormMode_t::BATCH_NORM)
                         .setxDesc(tensor(1, X_DIMS))
                         .setdyDesc(tensor(2, X_DIMS))
                         .setScale(tensor(3, CHANNEL_DIMS))
                         .setSavedMeanAndInvVar(tensor(4, CHANNEL_DIMS), tensor(5, CHANNEL_DIMS))
                         .setEpsilonTensor(scalar(6))
                         .setdxDesc(tensor(7, X_DIMS))
                         .setDScaleAndDBias(tensor(8, CHANNEL_DIMS), tensor(9, CHANNEL_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, LayerNormInference)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                         .setNormalizationMode(fe::NormMode_t::LAYER_NORM)
                         .setNormFwdPhase(fe::NormFwdPhase_t::INFERENCE)
                         .setxDesc(tensor(1, X_DIMS))
                         .setScaleAndBias(tensor(2, FEATURE_DIMS), tensor(3, FEATURE_DIMS))
                         .setEpsilonTensor(scalar(4))
                         .setyDesc(tensor(5, X_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, LayerNormBackward)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                         .setNormalizationMode(fe::NormMode_t::LAYER_NORM)
                         .setxDesc(tensor(1, X_DIMS))
                         .setdyDesc(tensor(2, X_DIMS))
                         .setScale(tensor(3, FEATURE_DIMS))
                         .setSavedMeanAndInvVar(tensor(4, SAMPLE_DIMS), tensor(5, SAMPLE_DIMS))
                         .setEpsilonTensor(scalar(6))
                         .setdxDesc(tensor(7, X_DIMS))
                         .setDScaleAndDBias(tensor(8, FEATURE_DIMS), tensor(9, FEATURE_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, RmsNormBackward)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                         .setNormalizationMode(fe::NormMode_t::RMS_NORM)
                         .setxDesc(tensor(1, X_DIMS))
                         .setdyDesc(tensor(2, X_DIMS))
                         .setScale(tensor(3, FEATURE_DIMS))
                         .setSavedInvVar(tensor(4, SAMPLE_DIMS))
                         .setdxDesc(tensor(5, X_DIMS))
                         .setDScale(tensor(6, FEATURE_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, OutOfOrderOpsSorted)
{
    const fe::Tensor convOut = virtualTensor(3, Y_DIMS);
    const fe::Tensor bias = tensor(4, {1, 8, 1, 1});
    const fe::Tensor biasOut = virtualTensor(5, Y_DIMS);
    std::vector<fe::Operation> ops;
    ops.emplace_back(pointwise(PointwiseMode_t::RELU_FWD, biasOut, nullptr, tensor(6, Y_DIMS)));
    ops.emplace_back(pointwise(PointwiseMode_t::ADD, convOut, &bias, biasOut));
    ops.emplace_back(convForward(tensor(1, X_DIMS), tensor(2, W_DIMS), convOut));
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, DisconnectedComponentsAccepted)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(relu(1, 2));
    ops.emplace_back(relu(3, 4));
    expectGraphBuilds(ops);
}

// Each builder call makes its own TENSOR descriptor, so equal tensors sharing a
// UID must collapse to one.
TEST_F(IntegrationCudnnShimV8BuilderLowering, EquivalentSameUidTensorsShared)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(relu(1, 2));
    ops.emplace_back(
        pointwise(PointwiseMode_t::TANH_FWD, tensor(2, X_DIMS), nullptr, tensor(3, X_DIMS)));
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, ConflictingSameUidTensorsRejected)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(relu(7, 2));
    ops.emplace_back(pointwise(
        PointwiseMode_t::RELU_FWD, tensor(7, POOLED_DIMS), nullptr, tensor(3, POOLED_DIMS)));
    expectGraphRejected(ops, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, MissingHandleRejected)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(relu(1, 2));
    fe::OperationGraphBuilder builder;
    builder.setOperationGraph(ops);
    EXPECT_EQ(buildStatus(builder), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, MissingOpsRejected)
{
    fe::OperationGraphBuilder builder;
    builder.setHandle(_handle);
    EXPECT_EQ(buildStatus(builder), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, DynamicShapeNotSupported)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(relu(1, 2));
    fe::OperationGraphBuilder builder;
    builder.setHandle(_handle).setOperationGraph(ops).setIsDynamicShapeEnabled(true);
    EXPECT_EQ(buildStatus(builder), CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(IntegrationCudnnShimV8BuilderLowering, OverrideShapeNotSupported)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(relu(1, 2));
    fe::OperationGraphBuilder builder;
    builder.setHandle(_handle).setOperationGraph(ops).setIsOverrideShapeEnabled(true);
    EXPECT_EQ(buildStatus(builder), CUDNN_STATUS_NOT_SUPPORTED);
}

} // namespace

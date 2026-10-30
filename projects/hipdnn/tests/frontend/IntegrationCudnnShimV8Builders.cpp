// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Builds cuDNN v8 operation graphs with the ported v0.x builders, including their
// always-set parameters (alpha/beta, the DOUBLE resample scales, the pointwise
// parameters), and lowers them against the real hipDNN backend. Graph lowering
// needs no device, so this suite does not skip on device-less hosts.

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

class IntegrationCudnnShimV8Builders : public DevicelessGraphFixture
{
};

TEST_F(IntegrationCudnnShimV8Builders, ConvolutionForward)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(convForward(tensor(1, X_DIMS), tensor(2, W_DIMS), tensor(3, Y_DIMS)));
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, ConvolutionBiasRelu)
{
    const fe::Tensor convOut = virtualTensor(3, Y_DIMS);
    const fe::Tensor bias = tensor(4, {1, 8, 1, 1});
    const fe::Tensor biasOut = virtualTensor(5, Y_DIMS);
    std::vector<fe::Operation> ops;
    ops.emplace_back(convForward(tensor(1, X_DIMS), tensor(2, W_DIMS), convOut));
    ops.emplace_back(pointwise(PointwiseMode_t::ADD, convOut, &bias, biasOut));
    ops.emplace_back(pointwise(PointwiseMode_t::RELU_FWD, biasOut, nullptr, tensor(6, Y_DIMS)));
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, MatmulBiasGeluTanh)
{
    const Dims cDims = {2, 4, 16};
    const fe::Tensor product = virtualTensor(3, cDims);
    const fe::Tensor bias = tensor(4, {1, 1, 16});
    const fe::Tensor biased = virtualTensor(5, cDims);
    const fe::MatMulDesc matmul = fe::MatMulDescBuilder().setComputeType(DataType_t::FLOAT).build();
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                         .setaMatDesc(tensor(1, {2, 4, 8}))
                         .setbMatDesc(tensor(2, {2, 8, 16}))
                         .setcMatDesc(product)
                         .setmatmulDesc(matmul)
                         .build());
    ops.emplace_back(pointwise(PointwiseMode_t::ADD, product, &bias, biased));
    ops.emplace_back(
        pointwise(PointwiseMode_t::GELU_APPROX_TANH_FWD, biased, nullptr, tensor(6, cDims)));
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, PointwiseChain)
{
    const fe::Tensor scale = tensor(2, CHANNEL_DIMS);
    const fe::Tensor scaled = virtualTensor(3, X_DIMS);
    const fe::Tensor shift = tensor(4, CHANNEL_DIMS);
    const fe::Tensor shifted = virtualTensor(5, X_DIMS);
    std::vector<fe::Operation> ops;
    ops.emplace_back(pointwise(PointwiseMode_t::MUL, tensor(1, X_DIMS), &scale, scaled));
    ops.emplace_back(pointwise(PointwiseMode_t::ADD, scaled, &shift, shifted));
    ops.emplace_back(pointwise(PointwiseMode_t::SIGMOID_FWD, shifted, nullptr, tensor(6, X_DIMS)));
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, Reduction)
{
    const fe::ReductionDesc reduction = fe::ReductionDescBuilder()
                                            .setComputeType(DataType_t::FLOAT)
                                            .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                            .build();
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR)
                         .setreductionDesc(reduction)
                         .setxDesc(tensor(1, X_DIMS))
                         .setyDesc(tensor(2, CHANNEL_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, ResampleForwardAverage)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                         .setResampleDesc(resampleDesc(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING))
                         .setxDesc(tensor(1, X_DIMS))
                         .setyDesc(tensor(2, POOLED_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, BatchNormTraining)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(
        fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
            .setNormalizationMode(fe::NormMode_t::BATCH_NORM)
            .setNormFwdPhase(fe::NormFwdPhase_t::TRAINING)
            .setxDesc(tensor(1, X_DIMS))
            .setScaleAndBias(tensor(2, CHANNEL_DIMS), tensor(3, CHANNEL_DIMS))
            .setEpsilonTensor(scalar(4))
            .setyDesc(tensor(5, X_DIMS))
            .setSavedMeanAndInvVar(tensor(6, CHANNEL_DIMS), tensor(7, CHANNEL_DIMS))
            .setPrevRunningMeanAndVar(tensor(8, CHANNEL_DIMS), tensor(9, CHANNEL_DIMS))
            .setExpDecayFactorTensor(scalar(10))
            .setNextRunningMeanAndVar(tensor(11, CHANNEL_DIMS), tensor(12, CHANNEL_DIMS))
            .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, LayerNormTraining)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                         .setNormalizationMode(fe::NormMode_t::LAYER_NORM)
                         .setNormFwdPhase(fe::NormFwdPhase_t::TRAINING)
                         .setxDesc(tensor(1, X_DIMS))
                         .setScaleAndBias(tensor(2, FEATURE_DIMS), tensor(3, FEATURE_DIMS))
                         .setEpsilonTensor(scalar(4))
                         .setyDesc(tensor(5, X_DIMS))
                         .setSavedMeanAndInvVar(tensor(6, SAMPLE_DIMS), tensor(7, SAMPLE_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, RmsNormTraining)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                         .setNormalizationMode(fe::NormMode_t::RMS_NORM)
                         .setNormFwdPhase(fe::NormFwdPhase_t::TRAINING)
                         .setxDesc(tensor(1, X_DIMS))
                         .setScale(tensor(2, FEATURE_DIMS))
                         .setEpsilonTensor(scalar(3))
                         .setyDesc(tensor(4, X_DIMS))
                         .setSavedInvVar(tensor(5, SAMPLE_DIMS))
                         .build());
    expectGraphBuilds(ops);
}

// cuDNN's layer-norm layout: [B*S, H, 1, 1] activations with [1, H, 1, 1] scale and bias.
TEST_F(IntegrationCudnnShimV8Builders, LayerNormCanonicalLayout)
{
    const Dims activation = {8, 16, 1, 1};
    const Dims feature = {1, 16, 1, 1};
    const Dims statistics = {8, 1, 1, 1};
    std::vector<fe::Operation> forward;
    forward.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                             .setNormalizationMode(fe::NormMode_t::LAYER_NORM)
                             .setNormFwdPhase(fe::NormFwdPhase_t::TRAINING)
                             .setxDesc(tensor(1, activation))
                             .setScaleAndBias(tensor(2, feature), tensor(3, feature))
                             .setEpsilonTensor(scalar(4))
                             .setyDesc(tensor(5, activation))
                             .setSavedMeanAndInvVar(tensor(6, statistics), tensor(7, statistics))
                             .build());
    expectGraphBuilds(forward);

    std::vector<fe::Operation> backward;
    backward.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                              .setNormalizationMode(fe::NormMode_t::LAYER_NORM)
                              .setxDesc(tensor(1, activation))
                              .setdyDesc(tensor(2, activation))
                              .setScale(tensor(3, feature))
                              .setSavedMeanAndInvVar(tensor(4, statistics), tensor(5, statistics))
                              .setdxDesc(tensor(6, activation))
                              .setDScaleAndDBias(tensor(7, feature), tensor(8, feature))
                              .build());
    expectGraphBuilds(backward);
}

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Builds cuDNN v8 operation graphs with the ported v0.x builders, whose
// always-set attributes (alpha/beta, the DOUBLE resample scales, the pointwise
// parameter sequence) the raw C API suites do not send, and finalizes them
// against the real hipDNN backend. Graph finalize needs no device, so this
// suite does not skip on device-less hosts.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>
#include <test_plugins/TestPluginConstants.hpp>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using fe::DataType_t;
using fe::DescriptorType_t;
using fe::PointwiseMode_t;

using Dims = std::vector<std::int64_t>;

const Dims X_DIMS = {2, 4, 8, 8};
const Dims W_DIMS = {8, 4, 3, 3};
const Dims Y_DIMS = {2, 8, 8, 8};
const Dims CHANNEL_DIMS = {1, 4, 1, 1};
const Dims SCALAR_DIMS = {1, 1, 1, 1};
// Per-sample statistics of a norm over C, H and W.
const Dims SAMPLE_DIMS = {2, 1, 1, 1};
const Dims FEATURE_DIMS = {1, 4, 8, 8};
const Dims POOLED_DIMS = {2, 4, 4, 4};

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

Dims packedStrides(const Dims& dims)
{
    Dims strides(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

fe::Tensor tensor(std::int64_t uid, const Dims& dims, bool isVirtual = false)
{
    const Dims strides = packedStrides(dims);
    return fe::TensorBuilder()
        .setDataType(DataType_t::FLOAT)
        .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
        .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
        .setId(uid)
        .setAlignment(16)
        .setVirtual(isVirtual)
        .build();
}

fe::Tensor scalar(std::int64_t uid)
{
    const Dims strides = packedStrides(SCALAR_DIMS);
    return fe::TensorBuilder()
        .setDataType(DataType_t::FLOAT)
        .setDim(static_cast<std::int64_t>(SCALAR_DIMS.size()), SCALAR_DIMS.data())
        .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
        .setId(uid)
        .setAlignment(16)
        .setByValue()
        .build();
}

fe::Operation convForward(const fe::Tensor& x, const fe::Tensor& w, const fe::Tensor& y)
{
    const std::array<std::int64_t, 2> ones = {1, 1};
    const fe::ConvDesc conv = fe::ConvDescBuilder()
                                  .setComputeType(DataType_t::FLOAT)
                                  .setMathMode(CUDNN_CROSS_CORRELATION)
                                  .setSpatialDimCount(2)
                                  .setPrePadding(2, ones.data())
                                  .setPostPadding(2, ones.data())
                                  .setDilation(2, ones.data())
                                  .setSpatialStride(2, ones.data())
                                  .build();
    return fe::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        .setxDesc(x)
        .setwDesc(w)
        .setyDesc(y)
        .setcDesc(conv)
        .build();
}

fe::Operation
    pointwise(PointwiseMode_t mode, const fe::Tensor& x, const fe::Tensor* b, const fe::Tensor& y)
{
    const fe::PointWiseDesc desc
        = fe::PointWiseDescBuilder().setMode(mode).setComputeType(DataType_t::FLOAT).build();
    fe::OperationBuilder builder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
    builder.setpwDesc(desc).setxDesc(x).setyDesc(y);
    if(b != nullptr)
    {
        builder.setbDesc(*b);
    }
    return builder.build();
}

class IntegrationCudnnShimV8Builders : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(cudnnCreate(&_handle), CUDNN_STATUS_SUCCESS);
        lastError();
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(cudnnDestroy(_handle), CUDNN_STATUS_SUCCESS);
        }
    }

    void expectGraphBuilds(const std::vector<fe::Operation>& ops)
    {
        try
        {
            fe::OperationGraphBuilder builder;
            builder.setHandle(_handle).setOperationGraph(ops);
            const fe::OperationGraph graph = builder.build();
            EXPECT_EQ(graph.get_status(), CUDNN_STATUS_SUCCESS) << graph.get_error();
        }
        catch(const fe::cudnnException& e)
        {
            ADD_FAILURE() << e.what();
        }
    }

    cudnnHandle_t _handle = nullptr;
};

TEST_F(IntegrationCudnnShimV8Builders, ConvolutionForward)
{
    std::vector<fe::Operation> ops;
    ops.emplace_back(convForward(tensor(1, X_DIMS), tensor(2, W_DIMS), tensor(3, Y_DIMS)));
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, ConvolutionBiasRelu)
{
    const fe::Tensor convOut = tensor(3, Y_DIMS, true);
    const fe::Tensor bias = tensor(4, {1, 8, 1, 1});
    const fe::Tensor biasOut = tensor(5, Y_DIMS, true);
    std::vector<fe::Operation> ops;
    ops.emplace_back(convForward(tensor(1, X_DIMS), tensor(2, W_DIMS), convOut));
    ops.emplace_back(pointwise(PointwiseMode_t::ADD, convOut, &bias, biasOut));
    ops.emplace_back(pointwise(PointwiseMode_t::RELU_FWD, biasOut, nullptr, tensor(6, Y_DIMS)));
    expectGraphBuilds(ops);
}

TEST_F(IntegrationCudnnShimV8Builders, MatmulBiasGeluTanh)
{
    const Dims cDims = {2, 4, 16};
    const fe::Tensor product = tensor(3, cDims, true);
    const fe::Tensor bias = tensor(4, {1, 1, 16});
    const fe::Tensor biased = tensor(5, cDims, true);
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
    const fe::Tensor scaled = tensor(3, X_DIMS, true);
    const fe::Tensor shift = tensor(4, CHANNEL_DIMS);
    const fe::Tensor shifted = tensor(5, X_DIMS, true);
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
    const std::array<std::int64_t, 2> window = {2, 2};
    const std::array<std::int64_t, 2> zeros = {0, 0};
    const fe::ResampleDesc_v8 resample
        = fe::ResampleDescBuilder_v8()
              .setResampleMode(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING)
              .setComputeType(DataType_t::FLOAT)
              .setPaddingMode(CUDNN_ZERO_PAD)
              .setSpatialDim(2, window.data())
              .setPrePadding(2, zeros.data())
              .setPostPadding(2, zeros.data())
              .setSpatialStride(2, window.data())
              .build();
    std::vector<fe::Operation> ops;
    ops.emplace_back(fe::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                         .setResampleDesc(resample)
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

} // namespace

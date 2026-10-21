// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Pins the shim's graph-lowering rules through the ported v0.x builders only,
// finalizing against the real hipDNN backend. Graph finalize needs no device,
// so this suite does not skip on device-less hosts.

#include <array>
#include <cstddef>
#include <cstdint>
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

Dims packedStrides(const Dims& dims)
{
    Dims strides(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

struct TensorOptions
{
    DataType_t dataType = DataType_t::FLOAT;
    bool isVirtual = false;
    bool isByValue = false;
};

fe::Tensor tensor(std::int64_t uid, const Dims& dims, const TensorOptions& options = {})
{
    const Dims strides = packedStrides(dims);
    return fe::TensorBuilder()
        .setDataType(options.dataType)
        .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
        .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
        .setId(uid)
        .setAlignment(16)
        .setVirtual(options.isVirtual)
        .setByValue(options.isByValue)
        .build();
}

fe::Tensor virtualTensor(std::int64_t uid, const Dims& dims)
{
    TensorOptions options;
    options.isVirtual = true;
    return tensor(uid, dims, options);
}

fe::Tensor scalar(std::int64_t uid)
{
    TensorOptions options;
    options.isByValue = true;
    return tensor(uid, SCALAR_DIMS, options);
}

fe::Tensor indexTensor(std::int64_t uid)
{
    TensorOptions options;
    options.dataType = DataType_t::INT32;
    return tensor(uid, POOLED_DIMS, options);
}

fe::ConvDesc convDesc()
{
    const std::array<std::int64_t, 2> ones = {1, 1};
    return fe::ConvDescBuilder()
        .setComputeType(DataType_t::FLOAT)
        .setMathMode(CUDNN_CROSS_CORRELATION)
        .setSpatialDimCount(2)
        .setPrePadding(2, ones.data())
        .setPostPadding(2, ones.data())
        .setDilation(2, ones.data())
        .setSpatialStride(2, ones.data())
        .build();
}

fe::Operation convForward(const fe::Tensor& x, const fe::Tensor& w, const fe::Tensor& y)
{
    return fe::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        .setxDesc(x)
        .setwDesc(w)
        .setyDesc(y)
        .setcDesc(convDesc())
        .build();
}

fe::PointWiseDesc pointwiseDesc(PointwiseMode_t mode)
{
    return fe::PointWiseDescBuilder().setMode(mode).setComputeType(DataType_t::FLOAT).build();
}

fe::Operation
    pointwise(PointwiseMode_t mode, const fe::Tensor& x, const fe::Tensor* b, const fe::Tensor& y)
{
    fe::OperationBuilder builder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
    builder.setpwDesc(pointwiseDesc(mode)).setxDesc(x).setyDesc(y);
    if(b != nullptr)
    {
        builder.setbDesc(*b);
    }
    return builder.build();
}

fe::Operation relu(std::int64_t xUid, std::int64_t yUid)
{
    return pointwise(
        PointwiseMode_t::RELU_FWD, tensor(xUid, X_DIMS), nullptr, tensor(yUid, X_DIMS));
}

// 2x2 window and stride, no padding; max pooling pads with -inf.
fe::ResampleDesc_v8 resampleDesc(cudnnResampleMode_t mode)
{
    const std::array<std::int64_t, 2> window = {2, 2};
    const std::array<std::int64_t, 2> zeros = {0, 0};
    return fe::ResampleDescBuilder_v8()
        .setResampleMode(mode)
        .setComputeType(DataType_t::FLOAT)
        .setPaddingMode(mode == CUDNN_RESAMPLE_MAXPOOL ? CUDNN_NEG_INF_PAD : CUDNN_ZERO_PAD)
        .setSpatialDim(2, window.data())
        .setPrePadding(2, zeros.data())
        .setPostPadding(2, zeros.data())
        .setSpatialStride(2, window.data())
        .build();
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

class IntegrationCudnnShimV8BuilderLowering : public ::testing::Test
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

    void expectGraphRejected(const std::vector<fe::Operation>& ops, cudnnStatus_t status)
    {
        fe::OperationGraphBuilder builder;
        builder.setHandle(_handle).setOperationGraph(ops);
        EXPECT_EQ(buildStatus(builder), status);
    }

    cudnnHandle_t _handle = nullptr;
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

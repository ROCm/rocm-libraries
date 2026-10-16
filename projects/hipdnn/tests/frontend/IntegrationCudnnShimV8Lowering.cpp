// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Finalizes cuDNN v8 operation graphs through the shim's C API against the real
// hipDNN backend, so the lowered ops pass real frontend-node and backend
// descriptor validation. Graph finalize needs no device, so this suite does not
// skip on device-less hosts.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <test_plugins/TestPluginConstants.hpp>

#include "CudnnShimV8GraphBuilder.hpp"

namespace
{
using hipdnn_shim_v8_test::GraphBuilder;
using hipdnn_shim_v8_test::lastError;
using hipdnn_shim_v8_test::TensorOptions;

const std::vector<std::int64_t> X_DIMS = {2, 4, 8, 8};
const std::vector<std::int64_t> CHANNEL_DIMS = {1, 4, 1, 1};
const std::vector<std::int64_t> SCALAR_DIMS = {1, 1, 1, 1};
// Per-sample statistics of a norm over C, H and W.
const std::vector<std::int64_t> SAMPLE_DIMS = {2, 1, 1, 1};
const std::vector<std::int64_t> FEATURE_DIMS = {1, 4, 8, 8};
const std::vector<std::int64_t> POOLED_DIMS = {2, 4, 4, 4};

class IntegrationCudnnShimV8Lowering : public ::testing::Test
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

    cudnnBackendDescriptor_t scalar(std::int64_t uid)
    {
        TensorOptions options;
        options.isByValue = true;
        return _builder.tensor(uid, SCALAR_DIMS, options);
    }

    cudnnBackendDescriptor_t pointwise(cudnnPointwiseMode_t mode,
                                       cudnnBackendDescriptor_t x,
                                       cudnnBackendDescriptor_t b,
                                       cudnnBackendDescriptor_t y)
    {
        std::vector<hipdnn_shim_v8_test::DescriptorSlot> slots
            = {{CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, _builder.pointwise(mode)},
               {CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x},
               {CUDNN_ATTR_OPERATION_POINTWISE_YDESC, y}};
        if(b != nullptr)
        {
            slots.emplace_back(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, b);
        }
        return _builder.finalizedOperation(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, slots);
    }

    cudnnBackendDescriptor_t convolutionForward(cudnnBackendDescriptor_t x,
                                                cudnnBackendDescriptor_t w,
                                                cudnnBackendDescriptor_t y)
    {
        return _builder.finalizedOperation(
            CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            {{CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, _builder.convolution(1, 1)},
             {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x},
             {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, w},
             {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, y}});
    }

    cudnnBackendDescriptor_t
        normForward(cudnnBackendNormMode_t mode,
                    cudnnBackendNormFwdPhase_t phase,
                    const std::vector<hipdnn_shim_v8_test::DescriptorSlot>& slots)
    {
        cudnnBackendDescriptor_t op
            = _builder.operation(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR, slots);
        GraphBuilder::set(op, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, CUDNN_TYPE_NORM_MODE, mode);
        GraphBuilder::set(
            op, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, CUDNN_TYPE_NORM_FWD_PHASE, phase);
        GraphBuilder::finalize(op);
        return op;
    }

    cudnnBackendDescriptor_t
        normBackward(cudnnBackendNormMode_t mode,
                     const std::vector<hipdnn_shim_v8_test::DescriptorSlot>& slots)
    {
        cudnnBackendDescriptor_t op
            = _builder.operation(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR, slots);
        GraphBuilder::set(op, CUDNN_ATTR_OPERATION_NORM_BWD_MODE, CUDNN_TYPE_NORM_MODE, mode);
        GraphBuilder::finalize(op);
        return op;
    }

    cudnnBackendDescriptor_t indexTensor(std::int64_t uid)
    {
        TensorOptions options;
        options.dataType = CUDNN_DATA_INT32;
        return _builder.tensor(uid, POOLED_DIMS, options);
    }

    void expectFinalized(const std::vector<cudnnBackendDescriptor_t>& ops)
    {
        cudnnBackendDescriptor_t graph = _builder.graph(_handle, ops);
        EXPECT_EQ(cudnnBackendFinalize(graph), CUDNN_STATUS_SUCCESS) << lastError();
    }

    static void expectRejected(cudnnBackendDescriptor_t graph,
                               cudnnStatus_t status,
                               const std::string& messagePart)
    {
        EXPECT_EQ(cudnnBackendFinalize(graph), status);
        const std::string message = lastError();
        EXPECT_NE(message.find(messagePart), std::string::npos) << message;
    }

    // Y = relu(X) on the shared activation shape.
    cudnnBackendDescriptor_t relu(std::int64_t xUid, std::int64_t yUid)
    {
        return pointwise(CUDNN_POINTWISE_RELU_FWD,
                         _builder.tensor(xUid, X_DIMS),
                         nullptr,
                         _builder.tensor(yUid, X_DIMS));
    }

    cudnnHandle_t _handle = nullptr;
    GraphBuilder _builder;
};

TEST_F(IntegrationCudnnShimV8Lowering, ConvolutionForward)
{
    expectFinalized({convolutionForward(_builder.tensor(1, X_DIMS),
                                        _builder.tensor(2, {8, 4, 3, 3}),
                                        _builder.tensor(3, {2, 8, 8, 8}))});
}

TEST_F(IntegrationCudnnShimV8Lowering, ConvolutionBackwardData)
{
    expectFinalized({_builder.finalizedOperation(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, _builder.convolution(1, 1)},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, _builder.tensor(1, {2, 8, 8, 8})},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, _builder.tensor(2, {8, 4, 3, 3})},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, _builder.tensor(3, X_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, ConvolutionBackwardFilter)
{
    expectFinalized({_builder.finalizedOperation(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, _builder.convolution(1, 1)},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, _builder.tensor(2, {2, 8, 8, 8})},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, _builder.tensor(3, {8, 4, 3, 3})}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, ConvolutionBiasRelu)
{
    const std::vector<std::int64_t> yDims = {2, 8, 8, 8};
    cudnnBackendDescriptor_t convOut = _builder.virtualTensor(3, yDims);
    cudnnBackendDescriptor_t biasOut = _builder.virtualTensor(5, yDims);
    expectFinalized(
        {convolutionForward(_builder.tensor(1, X_DIMS), _builder.tensor(2, {8, 4, 3, 3}), convOut),
         pointwise(CUDNN_POINTWISE_ADD, convOut, _builder.tensor(4, {1, 8, 1, 1}), biasOut),
         pointwise(CUDNN_POINTWISE_RELU_FWD, biasOut, nullptr, _builder.tensor(6, yDims))});
}

TEST_F(IntegrationCudnnShimV8Lowering, MatmulBiasGeluTanh)
{
    const std::vector<std::int64_t> cDims = {2, 4, 16};
    cudnnBackendDescriptor_t product = _builder.virtualTensor(3, cDims);
    cudnnBackendDescriptor_t biased = _builder.virtualTensor(5, cDims);
    expectFinalized(
        {_builder.finalizedOperation(
             CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
             {{CUDNN_ATTR_OPERATION_MATMUL_DESC, _builder.matmul()},
              {CUDNN_ATTR_OPERATION_MATMUL_ADESC, _builder.tensor(1, {2, 4, 8})},
              {CUDNN_ATTR_OPERATION_MATMUL_BDESC, _builder.tensor(2, {2, 8, 16})},
              {CUDNN_ATTR_OPERATION_MATMUL_CDESC, product}}),
         pointwise(CUDNN_POINTWISE_ADD, product, _builder.tensor(4, {1, 1, 16}), biased),
         pointwise(
             CUDNN_POINTWISE_GELU_APPROX_TANH_FWD, biased, nullptr, _builder.tensor(6, cDims))});
}

TEST_F(IntegrationCudnnShimV8Lowering, PointwiseChain)
{
    cudnnBackendDescriptor_t scaled = _builder.virtualTensor(3, X_DIMS);
    cudnnBackendDescriptor_t shifted = _builder.virtualTensor(5, X_DIMS);
    expectFinalized(
        {pointwise(CUDNN_POINTWISE_MUL,
                   _builder.tensor(1, X_DIMS),
                   _builder.tensor(2, CHANNEL_DIMS),
                   scaled),
         pointwise(CUDNN_POINTWISE_ADD, scaled, _builder.tensor(4, CHANNEL_DIMS), shifted),
         pointwise(CUDNN_POINTWISE_SIGMOID_FWD, shifted, nullptr, _builder.tensor(6, X_DIMS))});
}

TEST_F(IntegrationCudnnShimV8Lowering, PointwiseActivationBackward)
{
    expectFinalized({_builder.finalizedOperation(
        CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
          _builder.pointwise(CUDNN_POINTWISE_RELU_BWD)},
         {CUDNN_ATTR_OPERATION_POINTWISE_DYDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_POINTWISE_XDESC, _builder.tensor(2, X_DIMS)},
         {CUDNN_ATTR_OPERATION_POINTWISE_DXDESC, _builder.tensor(3, X_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, Reduction)
{
    expectFinalized({_builder.finalizedOperation(
        CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_REDUCTION_DESC, _builder.reduction(CUDNN_REDUCE_TENSOR_ADD)},
         {CUDNN_ATTR_OPERATION_REDUCTION_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_REDUCTION_YDESC, _builder.tensor(2, CHANNEL_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, ResampleForwardAverage)
{
    expectFinalized({_builder.finalizedOperation(
        CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC,
          _builder.resample(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, _builder.tensor(2, POOLED_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, ResampleForwardMaxWithIndex)
{
    expectFinalized({_builder.finalizedOperation(
        CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, _builder.resample(CUDNN_RESAMPLE_MAXPOOL)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, _builder.tensor(2, POOLED_DIMS)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC, indexTensor(3)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, ResampleBackwardAverage)
{
    expectFinalized({_builder.finalizedOperation(
        CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC,
          _builder.resample(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, _builder.tensor(1, POOLED_DIMS)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, _builder.tensor(2, X_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, ResampleBackwardMaxWithIndex)
{
    expectFinalized({_builder.finalizedOperation(
        CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC, _builder.resample(CUDNN_RESAMPLE_MAXPOOL)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, _builder.tensor(1, POOLED_DIMS)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC, indexTensor(2)},
         {CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, _builder.tensor(3, X_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, BatchNormTraining)
{
    expectFinalized({normForward(
        CUDNN_BATCH_NORM,
        CUDNN_NORM_FWD_TRAINING,
        {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, _builder.tensor(2, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, _builder.tensor(3, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(4)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, _builder.tensor(5, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, _builder.tensor(6, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, _builder.tensor(7, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC, _builder.tensor(8, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC, _builder.tensor(9, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC, scalar(10)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
          _builder.tensor(11, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
          _builder.tensor(12, CHANNEL_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, BatchNormInference)
{
    expectFinalized({normForward(
        CUDNN_BATCH_NORM,
        CUDNN_NORM_FWD_INFERENCE,
        {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, _builder.tensor(2, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, _builder.tensor(3, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, _builder.tensor(4, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, _builder.tensor(5, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(6)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, _builder.tensor(7, X_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, BatchNormBackward)
{
    expectFinalized({normBackward(
        CUDNN_BATCH_NORM,
        {{CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, _builder.tensor(2, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, _builder.tensor(3, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, _builder.tensor(4, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, _builder.tensor(5, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC, scalar(6)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, _builder.tensor(7, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, _builder.tensor(8, CHANNEL_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, _builder.tensor(9, CHANNEL_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, LayerNormTraining)
{
    expectFinalized({normForward(
        CUDNN_LAYER_NORM,
        CUDNN_NORM_FWD_TRAINING,
        {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, _builder.tensor(2, FEATURE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, _builder.tensor(3, FEATURE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(4)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, _builder.tensor(5, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, _builder.tensor(6, SAMPLE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, _builder.tensor(7, SAMPLE_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, LayerNormInference)
{
    expectFinalized(
        {normForward(CUDNN_LAYER_NORM,
                     CUDNN_NORM_FWD_INFERENCE,
                     {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, _builder.tensor(1, X_DIMS)},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, _builder.tensor(2, FEATURE_DIMS)},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, _builder.tensor(3, FEATURE_DIMS)},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(4)},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, _builder.tensor(5, X_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, LayerNormBackward)
{
    expectFinalized({normBackward(
        CUDNN_LAYER_NORM,
        {{CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, _builder.tensor(2, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, _builder.tensor(3, FEATURE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, _builder.tensor(4, SAMPLE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, _builder.tensor(5, SAMPLE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC, scalar(6)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, _builder.tensor(7, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, _builder.tensor(8, FEATURE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, _builder.tensor(9, FEATURE_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, RmsNormTraining)
{
    expectFinalized({normForward(
        CUDNN_RMS_NORM,
        CUDNN_NORM_FWD_TRAINING,
        {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, _builder.tensor(2, FEATURE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(3)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, _builder.tensor(4, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, _builder.tensor(5, SAMPLE_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, RmsNormBackward)
{
    expectFinalized({normBackward(
        CUDNN_RMS_NORM,
        {{CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, _builder.tensor(1, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, _builder.tensor(2, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, _builder.tensor(3, FEATURE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, _builder.tensor(4, SAMPLE_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, _builder.tensor(5, X_DIMS)},
         {CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, _builder.tensor(6, FEATURE_DIMS)}})});
}

TEST_F(IntegrationCudnnShimV8Lowering, OutOfOrderOpsSorted)
{
    const std::vector<std::int64_t> yDims = {2, 8, 8, 8};
    cudnnBackendDescriptor_t convOut = _builder.virtualTensor(3, yDims);
    cudnnBackendDescriptor_t biasOut = _builder.virtualTensor(5, yDims);
    cudnnBackendDescriptor_t conv
        = convolutionForward(_builder.tensor(1, X_DIMS), _builder.tensor(2, {8, 4, 3, 3}), convOut);
    cudnnBackendDescriptor_t add
        = pointwise(CUDNN_POINTWISE_ADD, convOut, _builder.tensor(4, {1, 8, 1, 1}), biasOut);
    cudnnBackendDescriptor_t activation
        = pointwise(CUDNN_POINTWISE_RELU_FWD, biasOut, nullptr, _builder.tensor(6, yDims));
    expectFinalized({activation, add, conv});
}

TEST_F(IntegrationCudnnShimV8Lowering, DisconnectedComponentsAccepted)
{
    expectFinalized({relu(1, 2), relu(3, 4)});
}

// Upstream builds a fresh TENSOR descriptor per reference, so equal tensors
// sharing a UID must collapse to one.
TEST_F(IntegrationCudnnShimV8Lowering, EquivalentSameUidTensorsShared)
{
    cudnnBackendDescriptor_t first = relu(1, 2);
    cudnnBackendDescriptor_t second = pointwise(
        CUDNN_POINTWISE_TANH_FWD, _builder.tensor(2, X_DIMS), nullptr, _builder.tensor(3, X_DIMS));
    expectFinalized({first, second});
}

TEST_F(IntegrationCudnnShimV8Lowering, ConflictingSameUidTensorsRejected)
{
    cudnnBackendDescriptor_t first = relu(7, 2);
    cudnnBackendDescriptor_t second = pointwise(CUDNN_POINTWISE_RELU_FWD,
                                                _builder.tensor(7, POOLED_DIMS),
                                                nullptr,
                                                _builder.tensor(3, POOLED_DIMS));
    expectRejected(_builder.graph(_handle, {first, second}), CUDNN_STATUS_BAD_PARAM, "UID 7");
}

TEST_F(IntegrationCudnnShimV8Lowering, MissingHandleRejected)
{
    expectRejected(_builder.graph(nullptr, {relu(1, 2)}),
                   CUDNN_STATUS_BAD_PARAM,
                   "CUDNN_ATTR_OPERATIONGRAPH_HANDLE");
}

TEST_F(IntegrationCudnnShimV8Lowering, MissingOpsRejected)
{
    cudnnBackendDescriptor_t graph = _builder.create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    GraphBuilder::set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, _handle);
    expectRejected(graph, CUDNN_STATUS_BAD_PARAM, "CUDNN_ATTR_OPERATIONGRAPH_OPS");
}

TEST_F(IntegrationCudnnShimV8Lowering, DynamicShapeNotSupported)
{
    cudnnBackendDescriptor_t graph = _builder.graph(_handle, {relu(1, 2)});
    GraphBuilder::set(
        graph, CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED, CUDNN_TYPE_BOOLEAN, true);
    expectRejected(
        graph, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED");
}

TEST_F(IntegrationCudnnShimV8Lowering, OverrideShapeNotSupported)
{
    cudnnBackendDescriptor_t graph = _builder.graph(_handle, {relu(1, 2)});
    GraphBuilder::set(
        graph, CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED, CUDNN_TYPE_BOOLEAN, true);
    expectRejected(
        graph, CUDNN_STATUS_NOT_SUPPORTED, "CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED");
}

TEST_F(IntegrationCudnnShimV8Lowering, EngineCountNeedsFinalizedGraph)
{
    cudnnBackendDescriptor_t graph = _builder.graph(_handle, {relu(1, 2)});
    std::int64_t count = -1;
    std::int64_t elementCount = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(graph,
                                       CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&count)),
              CUDNN_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(count, -1);
}

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// What each v0.x operation builder lowers to in hipDNN: operation type, slot tensors and parameters.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8BuilderFixture.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace v8fe = hipdnn_shim_test::v8fe;

using hipdnn_shim_test::Dims;
using v8fe::DataType_t;
using v8fe::DescriptorType_t;
using v8fe::NormFwdPhase_t;
using v8fe::NormMode_t;
using v8fe::PointwiseMode_t;

// hipDNN slot -> UID of the tensor it must hold; -1 means the slot is never set.
using SlotMap = std::vector<std::pair<hipdnnBackendAttributeName_t, std::int64_t>>;

const Dims ACTIVATION = {2, 3, 4, 4};
const Dims CHANNEL = {1, 3, 1, 1};
const Dims LAYER_SCALE = {1, 3, 4, 4};
const Dims PER_SAMPLE = {2, 1, 1, 1};
const Dims IMAGE = {1, 4, 8, 8};
const Dims FILTER = {8, 4, 3, 3};
const Dims FEATURES = {1, 8, 8, 8};
const Dims POOLED = {1, 4, 4, 4};

struct LoweringCase
{
    const char* label;
    std::function<v8fe::Operation()> build;
    hipdnnBackendDescriptorType_t type;
    hipdnnBackendAttributeName_t computeType;
    SlotMap slots;
    hipdnnDataType_t loweredComputeType = HIPDNN_DATA_FLOAT;
};

struct Pooling
{
    v8fe::ResampleMode_t mode;
    v8fe::PaddingMode_t padding;
    Dims window;
    Dims prePadding;
    Dims postPadding;
    Dims strides;
    DataType_t computeType = DataType_t::FLOAT;
};

class TestCudnnShimV8BuilderLowering : public hipdnn_shim_test::ShimV8BuilderFixture
{
protected:
    static v8fe::ConvDesc convolution(const Dims& prePadding,
                                      const Dims& postPadding,
                                      const Dims& strides,
                                      const Dims& dilations,
                                      cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION,
                                      DataType_t computeType = DataType_t::FLOAT)
    {
        const auto count = static_cast<std::int64_t>(prePadding.size());
        v8fe::ConvDescBuilder builder;
        return builder.setComputeType(computeType)
            .setMathMode(mode)
            .setSpatialDimCount(count)
            .setPrePadding(count, prePadding.data())
            .setPostPadding(count, postPadding.data())
            .setSpatialStride(count, strides.data())
            .setDilation(count, dilations.data())
            .build();
    }

    static v8fe::Operation convolutionBackwardData(const v8fe::Tensor& dy,
                                                   const v8fe::Tensor& w,
                                                   const v8fe::Tensor& dx,
                                                   const v8fe::ConvDesc& conv)
    {
        return v8fe::OperationBuilder(
                   DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
            .setdyDesc(dy)
            .setwDesc(w)
            .setdxDesc(dx)
            .setcDesc(conv)
            .build();
    }

    static v8fe::Operation convolutionBackwardFilter(const v8fe::Tensor& dy,
                                                     const v8fe::Tensor& x,
                                                     const v8fe::Tensor& dw,
                                                     const v8fe::ConvDesc& conv)
    {
        return v8fe::OperationBuilder(
                   DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
            .setdyDesc(dy)
            .setxDesc(x)
            .setdwDesc(dw)
            .setcDesc(conv)
            .build();
    }

    static v8fe::Operation pointwiseBackward(const v8fe::PointWiseDesc& desc,
                                             const v8fe::Tensor& dy,
                                             const v8fe::Tensor& x,
                                             const v8fe::Tensor& dx)
    {
        return v8fe::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
            .setpwDesc(desc)
            .setdyDesc(dy)
            .setxDesc(x)
            .setdxDesc(dx)
            .build();
    }

    static v8fe::Operation matmul(std::int64_t aUid,
                                  std::int64_t bUid,
                                  std::int64_t cUid,
                                  DataType_t computeType = DataType_t::FLOAT)
    {
        v8fe::MatMulDescBuilder descBuilder;
        const v8fe::MatMulDesc desc = descBuilder.setComputeType(computeType).build();
        return v8fe::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
            .setaMatDesc(tensor(aUid, {1, 4, 8}))
            .setbMatDesc(tensor(bUid, {1, 8, 5}))
            .setcMatDesc(tensor(cUid, {1, 4, 5}))
            .setmatmulDesc(desc)
            .build();
    }

    static v8fe::Operation reduction(std::int64_t xUid,
                                     std::int64_t yUid,
                                     std::optional<bool> isDeterministic,
                                     DataType_t computeType = DataType_t::FLOAT)
    {
        v8fe::ReductionDescBuilder descBuilder;
        descBuilder.setComputeType(computeType).setReductionOp(v8fe::ReductionMode_t::ADD);
        if(isDeterministic)
        {
            descBuilder.setIsDeterministic(*isDeterministic);
        }
        const v8fe::ReductionDesc desc = descBuilder.build();
        return v8fe::OperationBuilder(DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR)
            .setxDesc(tensor(xUid, {2, 3, 4, 5}))
            .setyDesc(tensor(yUid, {2, 3, 1, 1}))
            .setreductionDesc(desc)
            .build();
    }

    static v8fe::ResampleDesc_v8 resample(const Pooling& pooling)
    {
        const auto count = static_cast<std::int64_t>(pooling.window.size());
        v8fe::ResampleDescBuilder_v8 builder;
        return builder.setResampleMode(pooling.mode)
            .setComputeType(pooling.computeType)
            .setPaddingMode(pooling.padding)
            .setSpatialDim(count, pooling.window.data())
            .setPrePadding(count, pooling.prePadding.data())
            .setPostPadding(count, pooling.postPadding.data())
            .setSpatialStride(count, pooling.strides.data())
            .build();
    }

    // 2x2 window, stride 2, no padding: IMAGE -> POOLED.
    static v8fe::ResampleDesc_v8 maxPool()
    {
        return resample({v8fe::ResampleMode_t::MAXPOOL,
                         v8fe::PaddingMode_t::NEG_INF_PAD,
                         {2, 2},
                         {0, 0},
                         {0, 0},
                         {2, 2}});
    }

    static v8fe::ResampleDesc_v8 averagePool(DataType_t computeType = DataType_t::FLOAT)
    {
        return resample({v8fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING,
                         v8fe::PaddingMode_t::ZERO_PAD,
                         {2, 2},
                         {0, 0},
                         {0, 0},
                         {2, 2},
                         computeType});
    }

    static v8fe::Operation resampleForward(const v8fe::ResampleDesc_v8& desc,
                                           const v8fe::Tensor& x,
                                           const v8fe::Tensor& y,
                                           const v8fe::Tensor* index = nullptr)
    {
        v8fe::OperationBuilder builder(DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR);
        builder.setResampleDesc(desc).setxDesc(x).setyDesc(y);
        if(index != nullptr)
        {
            builder.setidxDesc(*index);
        }
        return builder.build();
    }

    static v8fe::Operation resampleBackward(const v8fe::ResampleDesc_v8& desc,
                                            const v8fe::Tensor& dy,
                                            const v8fe::Tensor& dx,
                                            const v8fe::Tensor* index = nullptr)
    {
        v8fe::OperationBuilder builder(DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR);
        builder.setResampleDesc(desc).setdyDesc(dy).setdxDesc(dx);
        if(index != nullptr)
        {
            builder.setidxDesc(*index);
        }
        return builder.build();
    }

    static v8fe::Tensor indexTensor(std::int64_t uid)
    {
        return tensor(uid, POOLED, false, DataType_t::INT32);
    }

    static v8fe::Operation
        layerNormBackward(const Dims& scale, const Dims& statistics, std::int64_t epsilonUid = -1)
    {
        v8fe::OperationBuilder builder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR);
        builder.setNormalizationMode(NormMode_t::LAYER_NORM)
            .setxDesc(tensor(1, ACTIVATION))
            .setdyDesc(tensor(2, ACTIVATION))
            .setScale(tensor(3, scale))
            .setSavedMeanAndInvVar(tensor(4, statistics), tensor(5, statistics))
            .setdxDesc(tensor(6, ACTIVATION))
            .setDScaleAndDBias(tensor(7, scale), tensor(8, scale));
        if(epsilonUid >= 0)
        {
            builder.setEpsilonTensor(scalar(epsilonUid));
        }
        return builder.build();
    }

    static v8fe::Operation layerNormInference(const Dims& scale)
    {
        return v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
            .setNormalizationMode(NormMode_t::LAYER_NORM)
            .setNormFwdPhase(NormFwdPhase_t::INFERENCE)
            .setxDesc(tensor(1, ACTIVATION))
            .setScaleAndBias(tensor(2, scale), tensor(3, scale))
            .setEpsilonTensor(scalar(4))
            .setyDesc(tensor(5, ACTIVATION))
            .build();
    }

    static v8fe::Operation rmsNormBackward(bool withDbias)
    {
        v8fe::OperationBuilder builder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR);
        builder.setNormalizationMode(NormMode_t::RMS_NORM)
            .setxDesc(tensor(1, ACTIVATION))
            .setdyDesc(tensor(2, ACTIVATION))
            .setScale(tensor(3, LAYER_SCALE))
            .setSavedInvVar(tensor(4, PER_SAMPLE))
            .setdxDesc(tensor(5, ACTIVATION))
            .setDScale(tensor(6, LAYER_SCALE));
        if(withDbias)
        {
            builder.setDBias(tensor(7, LAYER_SCALE));
        }
        return builder.build();
    }

    // One case per v0.x operation kind and norm mode/phase. Every tensor has
    // its own UID, so a swapped slot shows as a wrong UID.
    static std::vector<LoweringCase> loweringCases()
    {
        return {
            {"ConvolutionForward",
             [] {
                 return convolutionForward(
                     tensor(1, IMAGE), tensor(2, FILTER), tensor(3, FEATURES), convDesc());
             },
             HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
             HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
             {{HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, 1},
              {HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, 2},
              {HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, 3}}},
            {"ConvolutionBackwardData",
             [] {
                 return convolutionBackwardData(tensor(1, FEATURES),
                                                tensor(2, FILTER),
                                                tensor(3, IMAGE),
                                                convDesc(DataType_t::HALF));
             },
             HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
             HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
             {{HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, 1},
              {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, 2},
              {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, 3}},
             HIPDNN_DATA_HALF},
            {"ConvolutionBackwardFilter",
             [] {
                 return convolutionBackwardFilter(tensor(1, FEATURES),
                                                  tensor(2, IMAGE),
                                                  tensor(3, FILTER),
                                                  convDesc(DataType_t::BFLOAT16));
             },
             HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
             HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
             {{HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, 1},
              {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, 2},
              {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, 3}},
             HIPDNN_DATA_BFLOAT16},
            {"PointwiseMathUnary",
             [] {
                 return pointwise(pointwiseDesc(PointwiseMode_t::SQRT),
                                  tensor(1, ACTIVATION),
                                  tensor(2, ACTIVATION));
             },
             HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
             HIPDNN_ATTR_POINTWISE_MATH_PREC,
             {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, -1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, -1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 2}}},
            {"PointwiseMathBinary",
             [] {
                 const v8fe::Tensor b = tensor(2, CHANNEL);
                 return pointwise(pointwiseDesc(PointwiseMode_t::ADD, DataType_t::HALF),
                                  tensor(1, ACTIVATION),
                                  tensor(3, ACTIVATION),
                                  &b);
             },
             HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
             HIPDNN_ATTR_POINTWISE_MATH_PREC,
             {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, 2},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, -1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 3}},
             HIPDNN_DATA_HALF},
            {"PointwiseBinarySelect",
             [] {
                 const v8fe::Tensor b = tensor(2, ACTIVATION);
                 const v8fe::Tensor t = tensor(3, ACTIVATION, false, DataType_t::BOOLEAN);
                 return pointwise(pointwiseDesc(PointwiseMode_t::BINARY_SELECT),
                                  tensor(1, ACTIVATION),
                                  tensor(4, ACTIVATION),
                                  &b,
                                  &t);
             },
             HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
             HIPDNN_ATTR_POINTWISE_MATH_PREC,
             {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, 2},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, 3},
              {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 4}}},
            {"PointwiseIdentity",
             [] {
                 return pointwise(pointwiseDesc(PointwiseMode_t::IDENTITY),
                                  tensor(1, ACTIVATION),
                                  tensor(2, ACTIVATION));
             },
             HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
             HIPDNN_ATTR_POINTWISE_MATH_PREC,
             {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, -1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 2}}},
            {"PointwiseActivationForward",
             [] {
                 return pointwise(pointwiseDesc(PointwiseMode_t::RELU_FWD),
                                  tensor(1, ACTIVATION),
                                  tensor(2, ACTIVATION));
             },
             HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
             HIPDNN_ATTR_POINTWISE_MATH_PREC,
             {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, -1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 2}}},
            // Gradient first: IN_0 is dy, IN_1 is x.
            {"PointwiseActivationBackward",
             [] {
                 return pointwiseBackward(pointwiseDesc(PointwiseMode_t::RELU_BWD),
                                          tensor(1, ACTIVATION),
                                          tensor(2, ACTIVATION),
                                          tensor(3, ACTIVATION));
             },
             HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
             HIPDNN_ATTR_POINTWISE_MATH_PREC,
             {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, 2},
              {HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, -1},
              {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 3}}},
            {"Matmul",
             [] { return matmul(1, 2, 3, DataType_t::INT32); },
             HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
             HIPDNN_ATTR_MATMUL_COMP_TYPE,
             {{HIPDNN_ATTR_OPERATION_MATMUL_ADESC, 1},
              {HIPDNN_ATTR_OPERATION_MATMUL_BDESC, 2},
              {HIPDNN_ATTR_OPERATION_MATMUL_CDESC, 3}},
             HIPDNN_DATA_INT32},
            {"Reduction",
             [] { return reduction(1, 2, std::nullopt, DataType_t::DOUBLE); },
             HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
             HIPDNN_ATTR_REDUCTION_COMP_TYPE,
             {{HIPDNN_ATTR_OPERATION_REDUCTION_XDESC, 1},
              {HIPDNN_ATTR_OPERATION_REDUCTION_YDESC, 2}},
             HIPDNN_DATA_DOUBLE},
            {"ResampleForward",
             [] {
                 return resampleForward(
                     averagePool(DataType_t::HALF), tensor(1, IMAGE), tensor(2, POOLED));
             },
             HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
             HIPDNN_ATTR_RESAMPLE_COMP_TYPE,
             {{HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, 1},
              {HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, 2},
              {HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC, -1}},
             HIPDNN_DATA_HALF},
            {"ResampleBackward",
             [] {
                 return resampleBackward(
                     averagePool(DataType_t::BFLOAT16), tensor(1, POOLED), tensor(2, IMAGE));
             },
             HIPDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
             HIPDNN_ATTR_RESAMPLE_COMP_TYPE,
             {{HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, 1},
              {HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, 2},
              {HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC, -1}},
             HIPDNN_DATA_BFLOAT16},
            {"NormForwardBatchTraining",
             [] {
                 return v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                     .setNormalizationMode(NormMode_t::BATCH_NORM)
                     .setNormFwdPhase(NormFwdPhase_t::TRAINING)
                     .setxDesc(tensor(1, ACTIVATION))
                     .setScaleAndBias(tensor(2, CHANNEL), tensor(3, CHANNEL))
                     .setEpsilonTensor(scalar(4))
                     .setyDesc(tensor(5, ACTIVATION))
                     .setSavedMeanAndInvVar(tensor(6, CHANNEL), tensor(7, CHANNEL))
                     .build();
             },
             HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT,
             HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, 2},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, 3},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, 4},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, 5},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, 6},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT, 7}}},
            {"NormForwardBatchInference",
             [] {
                 return v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                     .setNormalizationMode(NormMode_t::BATCH_NORM)
                     .setNormFwdPhase(NormFwdPhase_t::INFERENCE)
                     .setxDesc(tensor(1, ACTIVATION))
                     .setSavedMeanAndInvVar(tensor(2, CHANNEL), tensor(3, CHANNEL))
                     .setScaleAndBias(tensor(4, CHANNEL), tensor(5, CHANNEL))
                     .setyDesc(tensor(6, ACTIVATION))
                     .build();
             },
             HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT,
             HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT, 2},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT, 3},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT, 4},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT, 5},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT, 6}}},
            {"NormForwardLayerTraining",
             [] {
                 return v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                     .setNormalizationMode(NormMode_t::LAYER_NORM)
                     .setNormFwdPhase(NormFwdPhase_t::TRAINING)
                     .setxDesc(tensor(1, ACTIVATION))
                     .setScaleAndBias(tensor(2, LAYER_SCALE), tensor(3, LAYER_SCALE))
                     .setEpsilonTensor(scalar(4))
                     .setyDesc(tensor(5, ACTIVATION))
                     .setSavedMeanAndInvVar(tensor(6, PER_SAMPLE), tensor(7, PER_SAMPLE))
                     .build();
             },
             HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT,
             HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT, 2},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT, 3},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT, 4},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT, 5},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT, 6},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT, 7}}},
            {"NormForwardLayerInference",
             [] { return layerNormInference(LAYER_SCALE); },
             HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT,
             HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT, 2},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT, 3},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT, 4},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT, 5},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT, -1},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT, -1}}},
            {"NormForwardRmsTraining",
             [] {
                 return v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                     .setNormalizationMode(NormMode_t::RMS_NORM)
                     .setNormFwdPhase(NormFwdPhase_t::TRAINING)
                     .setxDesc(tensor(1, ACTIVATION))
                     .setScale(tensor(2, LAYER_SCALE))
                     .setEpsilonTensor(scalar(3))
                     .setyDesc(tensor(4, ACTIVATION))
                     .setSavedInvVar(tensor(5, PER_SAMPLE))
                     .setBias(tensor(6, LAYER_SCALE))
                     .build();
             },
             HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT,
             HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT, 2},
              {HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT, 3},
              {HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT, 4},
              {HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT, 5},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT, 6}}},
            {"NormForwardRmsInference",
             [] {
                 return v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                     .setNormalizationMode(NormMode_t::RMS_NORM)
                     .setNormFwdPhase(NormFwdPhase_t::INFERENCE)
                     .setxDesc(tensor(1, ACTIVATION))
                     .setScale(tensor(2, LAYER_SCALE))
                     .setEpsilonTensor(scalar(3))
                     .setyDesc(tensor(4, ACTIVATION))
                     .build();
             },
             HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT,
             HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT, 2},
              {HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT, 3},
              {HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT, 4},
              {HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT, -1},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT, -1}}},
            {"NormBackwardBatch",
             [] {
                 return v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                     .setNormalizationMode(NormMode_t::BATCH_NORM)
                     .setxDesc(tensor(1, ACTIVATION))
                     .setdyDesc(tensor(2, ACTIVATION))
                     .setScale(tensor(3, CHANNEL))
                     .setSavedMeanAndInvVar(tensor(4, CHANNEL), tensor(5, CHANNEL))
                     .setdxDesc(tensor(6, ACTIVATION))
                     .setDScaleAndDBias(tensor(7, CHANNEL), tensor(8, CHANNEL))
                     .build();
             },
             HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT,
             HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, 2},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, 3},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, 4},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, 5},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, 6},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, 7},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, 8}}},
            {"NormBackwardLayer",
             [] { return layerNormBackward(LAYER_SCALE, PER_SAMPLE); },
             HIPDNN_BACKEND_OPERATION_LAYERNORM_BACKWARD_DESCRIPTOR_EXT,
             HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY_EXT, 2},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE_EXT, 3},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN_EXT, 4},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE_EXT, 5},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX_EXT, 6},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE_EXT, 7},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS_EXT, 8},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON_EXT, -1}}},
            {"NormBackwardRms",
             [] { return rmsNormBackward(true); },
             HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT,
             HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT, 2},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT, 3},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT, 4},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT, 5},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT, 6},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT, 7}}},
        };
    }

    // Lowers a one-operation graph and returns its hipDNN operation.
    hipdnnBackendDescriptor_t lowered(const v8fe::Operation& op) const
    {
        graph({&op});
        const std::vector<hipdnnBackendDescriptor_t> ops = graphOps();
        EXPECT_EQ(ops.size(), 1U);
        return ops.empty() ? nullptr : ops.front();
    }

    // Each slot holds the one hipDNN tensor lowered for its UID, or is never set.
    void expectSlots(hipdnnBackendDescriptor_t op, const SlotMap& slots) const
    {
        for(const auto& [slot, uid] : slots)
        {
            SCOPED_TRACE("hipDNN attribute " + std::to_string(slot));
            if(uid < 0)
            {
                EXPECT_FALSE(callOn(op, slot));
                continue;
            }
            EXPECT_EQ(slotUid(op, slot), uid);
            EXPECT_EQ(slotTensor(op, slot), hipdnnTensor(uid));
        }
    }

    bool lowersUid(std::int64_t uid) const
    {
        for(const auto& call : callsNamed(HIPDNN_ATTR_TENSOR_UNIQUE_ID))
        {
            if(call.value<std::int64_t>() == uid)
            {
                return true;
            }
        }
        return false;
    }
};

TEST_F(TestCudnnShimV8BuilderLowering, TargetPerOperation)
{
    const std::vector<LoweringCase> cases = loweringCases();
    ASSERT_EQ(cases.size(), 22U);
    std::set<hipdnnBackendDescriptorType_t> seen;
    for(const LoweringCase& item : cases)
    {
        SCOPED_TRACE(item.label);
        const v8fe::Operation op = item.build();
        hipdnnBackendDescriptor_t hipdnnOp = lowered(op);
        ASSERT_NE(hipdnnOp, nullptr);
        EXPECT_EQ(typeOf(hipdnnOp), item.type);
        seen.insert(item.type);
    }
    EXPECT_EQ(seen.size(), 15U);
}

TEST_F(TestCudnnShimV8BuilderLowering, SlotsMapByUid)
{
    for(const LoweringCase& item : loweringCases())
    {
        SCOPED_TRACE(item.label);
        const v8fe::Operation op = item.build();
        hipdnnBackendDescriptor_t hipdnnOp = lowered(op);
        ASSERT_NE(hipdnnOp, nullptr);
        expectSlots(hipdnnOp, item.slots);
    }
}

TEST_F(TestCudnnShimV8BuilderLowering, ComputeTypeReachesEveryOperation)
{
    // Norm builders take no compute type; the lowering supplies FLOAT.
    for(const LoweringCase& item : loweringCases())
    {
        SCOPED_TRACE(item.label);
        const v8fe::Operation op = item.build();
        hipdnnBackendDescriptor_t hipdnnOp = lowered(op);
        ASSERT_NE(hipdnnOp, nullptr);
        const auto computeType = callOn(hipdnnOp, item.computeType);
        ASSERT_TRUE(computeType);
        EXPECT_EQ(computeType->value<hipdnnDataType_t>(), item.loweredComputeType);
    }
}

TEST_F(TestCudnnShimV8BuilderLowering, UnsupportedOperationHasNoTarget)
{
    v8fe::RngDescBuilder descBuilder;
    const v8fe::RngDesc desc = descBuilder.setRngDistribution(v8fe::RngDistribution_t::BERNOULLI)
                                   .setBernoulliDistProbability(0.5)
                                   .build();
    v8fe::OperationBuilder rng(DescriptorType_t::OPERATION_RNG_DESCRIPTOR);
    rng.setyDesc(tensor(1, ACTIVATION)).setRngDesc(desc).setSeed(7);
    // Whether the operation or the graph build rejects it is not the rule.
    const hipdnn_shim_test::BuildFailure failure = hipdnn_shim_test::failureOf([&] {
        const v8fe::Operation op = rng.build();
        graph({&op});
    });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimV8BuilderLowering, ConvolutionParametersReachEveryKind)
{
    // Pre and post paddings differ per axis but keep the per-axis sums, so the
    // output is [1,8,4,6]. True convolution is not hipDNN's default mode.
    const Dims output = {1, 8, 4, 6};
    const v8fe::ConvDesc conv = convolution({1, 0}, {1, 2}, {2, 1}, {1, 2}, CUDNN_CONVOLUTION);
    const v8fe::Operation forward
        = convolutionForward(tensor(1, IMAGE), tensor(2, FILTER), tensor(3, output), conv);
    const v8fe::Operation data
        = convolutionBackwardData(tensor(4, output), tensor(5, FILTER), tensor(6, IMAGE), conv);
    const v8fe::Operation filter
        = convolutionBackwardFilter(tensor(7, output), tensor(8, IMAGE), tensor(9, FILTER), conv);
    graph({&forward, &data, &filter});

    const std::array<std::pair<hipdnnBackendDescriptorType_t, SlotMap>, 3> kinds = {{
        {HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
         {{HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, 1},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, 2},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, 3}}},
        {HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
         {{HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, 4},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, 5},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, 6}}},
        {HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
         {{HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, 7},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, 8},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, 9}}},
    }};
    for(const auto& [type, slots] : kinds)
    {
        SCOPED_TRACE("hipDNN descriptor type " + std::to_string(type));
        hipdnnBackendDescriptor_t op = onlyCreated(type);
        ASSERT_NE(op, nullptr);
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS).values<std::int64_t>(),
                  (Dims{1, 0}));
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS).values<std::int64_t>(),
                  (Dims{1, 2}));
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES).values<std::int64_t>(),
                  (Dims{2, 1}));
        EXPECT_EQ(attributeOn(op, HIPDNN_ATTR_CONVOLUTION_DILATIONS).values<std::int64_t>(),
                  (Dims{1, 2}));
        EXPECT_EQ(
            attributeOn(op, HIPDNN_ATTR_CONVOLUTION_CONV_MODE).value<hipdnnConvolutionMode_t>(),
            HIPDNN_CONVOLUTION);
        expectSlots(op, slots);
    }

    hipdnnBackendDescriptor_t graphDescriptor
        = onlyCreated(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    EXPECT_EQ(
        attributeOn(graphDescriptor, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE).value<hipdnnHandle_t>(),
        _handle);
}

TEST_F(TestCudnnShimV8BuilderLowering, PointwiseModeAndSlotsPerClass)
{
    struct ForwardCase
    {
        PointwiseMode_t mode;
        hipdnnPointwiseMode_t lowered;
        std::int64_t ports;
    };
    for(const ForwardCase& item :
        {ForwardCase{PointwiseMode_t::SQRT, HIPDNN_POINTWISE_SQRT, 2},
         ForwardCase{PointwiseMode_t::ERF, HIPDNN_POINTWISE_ERF, 2},
         ForwardCase{PointwiseMode_t::ADD, HIPDNN_POINTWISE_ADD, 3},
         ForwardCase{PointwiseMode_t::BINARY_SELECT, HIPDNN_POINTWISE_BINARY_SELECT, 4},
         ForwardCase{PointwiseMode_t::IDENTITY, HIPDNN_POINTWISE_IDENTITY, 2},
         ForwardCase{PointwiseMode_t::EXP, HIPDNN_POINTWISE_EXP, 2}})
    {
        SCOPED_TRACE(static_cast<int>(item.lowered));
        const v8fe::Tensor b = tensor(11, ACTIVATION);
        const v8fe::Tensor t = tensor(12, ACTIVATION, false, DataType_t::BOOLEAN);
        // The builder ignores b and t beyond the mode's port count.
        const v8fe::Operation op = pointwise(
            pointwiseDesc(item.mode), tensor(10, ACTIVATION), tensor(13, ACTIVATION), &b, &t);
        hipdnnBackendDescriptor_t hipdnnOp = lowered(op);
        ASSERT_NE(hipdnnOp, nullptr);
        EXPECT_EQ(attributeOn(hipdnnOp, HIPDNN_ATTR_POINTWISE_MODE).value<hipdnnPointwiseMode_t>(),
                  item.lowered);
        EXPECT_EQ(attributeOn(hipdnnOp, HIPDNN_ATTR_POINTWISE_MATH_PREC).value<hipdnnDataType_t>(),
                  HIPDNN_DATA_FLOAT);
        expectSlots(hipdnnOp,
                    {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 10},
                     {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, item.ports >= 3 ? 11 : -1},
                     {HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, item.ports == 4 ? 12 : -1},
                     {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 13}});
    }

    const v8fe::Operation reluBackward = pointwiseBackward(pointwiseDesc(PointwiseMode_t::RELU_BWD),
                                                           tensor(20, ACTIVATION),
                                                           tensor(21, ACTIVATION),
                                                           tensor(22, ACTIVATION));
    hipdnnBackendDescriptor_t backward = lowered(reluBackward);
    ASSERT_NE(backward, nullptr);
    EXPECT_EQ(attributeOn(backward, HIPDNN_ATTR_POINTWISE_MODE).value<hipdnnPointwiseMode_t>(),
              HIPDNN_POINTWISE_RELU_BWD);
    expectSlots(backward,
                {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 20},
                 {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, 21},
                 {HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, -1},
                 {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 22}});
}

TEST_F(TestCudnnShimV8BuilderLowering, PointwiseParametersSentOnlyWhenNeeded)
{
    constexpr float FLOAT_MAX = std::numeric_limits<float>::max();
    struct ParameterCase
    {
        const char* label;
        PointwiseMode_t mode;
        std::function<void(v8fe::PointWiseDescBuilder&)> configure;
        // In the order of `parameters` below.
        std::vector<std::pair<hipdnnBackendAttributeName_t, float>> sent;
    };
    const std::vector<ParameterCase> cases = {
        {"relu defaults: nothing sent",
         PointwiseMode_t::RELU_FWD,
         [](v8fe::PointWiseDescBuilder&) {},
         {}},
        {"relu lower clip below float range saturates",
         PointwiseMode_t::RELU_FWD,
         [](v8fe::PointWiseDescBuilder& builder) { builder.setReluLowerClip(-1e300); },
         {{HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, -FLOAT_MAX}}},
        {"relu upper clip alone",
         PointwiseMode_t::RELU_FWD,
         [](v8fe::PointWiseDescBuilder& builder) { builder.setReluUpperClip(6.0); },
         {{HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP, 6.0F}}},
        {"relu upper clip beyond float range clips nothing",
         PointwiseMode_t::RELU_FWD,
         [](v8fe::PointWiseDescBuilder& builder) { builder.setReluUpperClip(1e300); },
         {}},
        {"relu slope alone",
         PointwiseMode_t::RELU_FWD,
         [](v8fe::PointWiseDescBuilder& builder) { builder.setReluLowerClipSlope(0.25); },
         {{HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, 0.25F}}},
        {"relu slope keeps a non-default lower clip",
         PointwiseMode_t::RELU_FWD,
         [](v8fe::PointWiseDescBuilder& builder) {
             builder.setReluLowerClip(0.5).setReluLowerClipSlope(0.25);
         },
         {{HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP, 0.5F},
          {HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, 0.25F}}},
        {"elu default alpha", PointwiseMode_t::ELU_FWD, [](v8fe::PointWiseDescBuilder&) {}, {}},
        {"elu alpha",
         PointwiseMode_t::ELU_FWD,
         [](v8fe::PointWiseDescBuilder& builder) { builder.setEluAlpha(0.5); },
         {{HIPDNN_ATTR_POINTWISE_ELU_ALPHA, 0.5F}}},
        {"elu alpha beyond float range saturates",
         PointwiseMode_t::ELU_FWD,
         [](v8fe::PointWiseDescBuilder& builder) { builder.setEluAlpha(1e300); },
         {{HIPDNN_ATTR_POINTWISE_ELU_ALPHA, FLOAT_MAX}}},
        {"softplus beta",
         PointwiseMode_t::SOFTPLUS_FWD,
         [](v8fe::PointWiseDescBuilder& builder) { builder.setSoftplusBeta(2.5); },
         {{HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA, 2.5F}}},
        {"swish beta",
         PointwiseMode_t::SWISH_FWD,
         [](v8fe::PointWiseDescBuilder& builder) { builder.setSwishBeta(3.5); },
         {{HIPDNN_ATTR_POINTWISE_SWISH_BETA, 3.5F}}},
    };
    const std::array<hipdnnBackendAttributeName_t, 7> parameters
        = {HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
           HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
           HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
           HIPDNN_ATTR_POINTWISE_ELU_ALPHA,
           HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
           HIPDNN_ATTR_POINTWISE_SWISH_BETA,
           HIPDNN_ATTR_POINTWISE_AXIS};
    for(const ParameterCase& item : cases)
    {
        SCOPED_TRACE(item.label);
        v8fe::PointWiseDescBuilder descBuilder;
        descBuilder.setMode(item.mode).setComputeType(DataType_t::FLOAT);
        item.configure(descBuilder);
        const v8fe::PointWiseDesc desc = descBuilder.build();
        const v8fe::Operation op = pointwise(desc, tensor(1, ACTIVATION), tensor(2, ACTIVATION));
        hipdnnBackendDescriptor_t hipdnnOp = lowered(op);
        ASSERT_NE(hipdnnOp, nullptr);

        std::vector<hipdnnBackendAttributeName_t> sent;
        for(const hipdnnBackendAttributeName_t name : parameters)
        {
            if(callOn(hipdnnOp, name))
            {
                sent.push_back(name);
            }
        }
        std::vector<hipdnnBackendAttributeName_t> expected;
        for(const auto& [name, value] : item.sent)
        {
            expected.push_back(name);
            const auto call = callOn(hipdnnOp, name);
            ASSERT_TRUE(call) << "hipDNN attribute " << name;
            EXPECT_EQ(call->type, HIPDNN_TYPE_FLOAT);
            EXPECT_EQ(call->value<float>(), value);
        }
        EXPECT_EQ(sent, expected);
    }
}

TEST_F(TestCudnnShimV8BuilderLowering, PointwiseAxisFollowsGenIndex)
{
    for(const std::optional<std::int64_t> axis :
        {std::optional<std::int64_t>{2}, std::optional<std::int64_t>{}})
    {
        SCOPED_TRACE(axis ? "axis 2" : "default axis");
        v8fe::PointWiseDescBuilder descBuilder;
        descBuilder.setMode(PointwiseMode_t::GEN_INDEX).setComputeType(DataType_t::FLOAT);
        if(axis)
        {
            descBuilder.setAxis(*axis);
        }
        const v8fe::PointWiseDesc desc = descBuilder.build();
        const v8fe::Operation op = pointwise(
            desc, tensor(1, ACTIVATION), tensor(2, ACTIVATION, false, DataType_t::INT32));
        hipdnnBackendDescriptor_t hipdnnOp = lowered(op);
        ASSERT_NE(hipdnnOp, nullptr);
        const auto sent = callOn(hipdnnOp, HIPDNN_ATTR_POINTWISE_AXIS);
        ASSERT_EQ(sent.has_value(), axis.has_value());
        if(axis)
        {
            EXPECT_EQ(sent->value<std::int64_t>(), *axis);
        }
    }
}

TEST_F(TestCudnnShimV8BuilderLowering, LayerNormNormalizedDimCountFromScale)
{
    struct DimCase
    {
        Dims scale;
        Dims statistics;
        std::int64_t expected;
    };
    // A full-rank scale normalizes from its first non-1 dim on, else the scale's rank. [1, C, 1, 1]
    // is cuDNN's layer-norm layout ([B*S, H, 1, 1] activations); hipDNN's backward node rejects
    // any count under 3 for it.
    for(const DimCase& item : {DimCase{{1, 1, 1, 4}, {2, 3, 4, 1}, 1},
                               DimCase{{1, 1, 4, 4}, {2, 3, 1, 1}, 2},
                               DimCase{{1, 3, 4, 4}, {2, 1, 1, 1}, 3},
                               DimCase{{1, 3, 1, 1}, {2, 1, 1, 1}, 3},
                               DimCase{{3, 4, 4}, {2}, 3}})
    {
        SCOPED_TRACE(std::to_string(item.scale.size()) + "-D scale, "
                     + std::to_string(item.expected) + " normalized dims");
        const v8fe::Operation forward = layerNormInference(item.scale);
        hipdnnBackendDescriptor_t forwardOp = lowered(forward);
        ASSERT_NE(forwardOp, nullptr);
        const auto forwardCount
            = callOn(forwardOp, HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT);
        ASSERT_TRUE(forwardCount);
        EXPECT_EQ(forwardCount->value<std::int64_t>(), item.expected);

        const v8fe::Operation backward = layerNormBackward(item.scale, item.statistics);
        hipdnnBackendDescriptor_t backwardOp = lowered(backward);
        ASSERT_NE(backwardOp, nullptr);
        const auto backwardCount
            = callOn(backwardOp, HIPDNN_ATTR_LAYERNORM_BACKWARD_NORMALIZED_DIM_COUNT_EXT);
        ASSERT_TRUE(backwardCount);
        EXPECT_EQ(backwardCount->value<std::int64_t>(), item.expected);
    }
}

TEST_F(TestCudnnShimV8BuilderLowering, NormForwardPhaseReachesOperation)
{
    struct PhaseCase
    {
        const char* label;
        NormMode_t mode;
        NormFwdPhase_t phase;
        hipdnnBackendAttributeName_t attribute;
        hipdnnNormFwdPhase_t lowered;
    };
    for(const PhaseCase& item : {PhaseCase{"layer training",
                                           NormMode_t::LAYER_NORM,
                                           NormFwdPhase_t::TRAINING,
                                           HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                                           HIPDNN_NORM_FWD_TRAINING},
                                 PhaseCase{"layer inference",
                                           NormMode_t::LAYER_NORM,
                                           NormFwdPhase_t::INFERENCE,
                                           HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                                           HIPDNN_NORM_FWD_INFERENCE},
                                 PhaseCase{"rms training",
                                           NormMode_t::RMS_NORM,
                                           NormFwdPhase_t::TRAINING,
                                           HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                                           HIPDNN_NORM_FWD_TRAINING},
                                 PhaseCase{"rms inference",
                                           NormMode_t::RMS_NORM,
                                           NormFwdPhase_t::INFERENCE,
                                           HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                                           HIPDNN_NORM_FWD_INFERENCE}})
    {
        SCOPED_TRACE(item.label);
        v8fe::OperationBuilder builder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);
        builder.setNormalizationMode(item.mode)
            .setNormFwdPhase(item.phase)
            .setxDesc(tensor(1, ACTIVATION))
            .setScale(tensor(2, LAYER_SCALE))
            .setEpsilonTensor(scalar(3))
            .setyDesc(tensor(4, ACTIVATION));
        if(item.mode == NormMode_t::LAYER_NORM)
        {
            builder.setBias(tensor(5, LAYER_SCALE));
        }
        const v8fe::Operation op = builder.build();
        hipdnnBackendDescriptor_t hipdnnOp = lowered(op);
        ASSERT_NE(hipdnnOp, nullptr);
        const auto phase = callOn(hipdnnOp, item.attribute);
        ASSERT_TRUE(phase);
        EXPECT_EQ(phase->value<hipdnnNormFwdPhase_t>(), item.lowered);
    }
}

TEST_F(TestCudnnShimV8BuilderLowering, BatchNormDropsEpsilon)
{
    constexpr std::int64_t INFERENCE_EPSILON = 90;
    const v8fe::Operation inference
        = v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
              .setNormalizationMode(NormMode_t::BATCH_NORM)
              .setNormFwdPhase(NormFwdPhase_t::INFERENCE)
              .setxDesc(tensor(1, ACTIVATION))
              .setSavedMeanAndInvVar(tensor(2, CHANNEL), tensor(3, CHANNEL))
              .setScaleAndBias(tensor(4, CHANNEL), tensor(5, CHANNEL))
              .setEpsilonTensor(scalar(INFERENCE_EPSILON))
              .setyDesc(tensor(6, ACTIVATION))
              .build();
    hipdnnBackendDescriptor_t inferenceOp = lowered(inference);
    ASSERT_NE(inferenceOp, nullptr);
    EXPECT_EQ(typeOf(inferenceOp), HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT);
    EXPECT_FALSE(lowersUid(INFERENCE_EPSILON));
    expectSlots(inferenceOp,
                {{HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT, 2},
                 {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT, 3}});

    constexpr std::int64_t BACKWARD_EPSILON = 91;
    const v8fe::Operation batchBackward
        = v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
              .setNormalizationMode(NormMode_t::BATCH_NORM)
              .setxDesc(tensor(1, ACTIVATION))
              .setdyDesc(tensor(2, ACTIVATION))
              .setScale(tensor(3, CHANNEL))
              .setSavedMeanAndInvVar(tensor(4, CHANNEL), tensor(5, CHANNEL))
              .setEpsilonTensor(scalar(BACKWARD_EPSILON))
              .setdxDesc(tensor(6, ACTIVATION))
              .setDScaleAndDBias(tensor(7, CHANNEL), tensor(8, CHANNEL))
              .build();
    ASSERT_NE(lowered(batchBackward), nullptr);
    EXPECT_FALSE(lowersUid(BACKWARD_EPSILON));

    constexpr std::int64_t LAYER_EPSILON = 92;
    const v8fe::Operation layerBackward = layerNormBackward(LAYER_SCALE, PER_SAMPLE, LAYER_EPSILON);
    hipdnnBackendDescriptor_t layerOp = lowered(layerBackward);
    ASSERT_NE(layerOp, nullptr);
    expectSlots(layerOp, {{HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON_EXT, LAYER_EPSILON}});
}

TEST_F(TestCudnnShimV8BuilderLowering, BatchNormRunningStatsBecomeMomentumUpdate)
{
    const v8fe::Operation op
        = v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
              .setNormalizationMode(NormMode_t::BATCH_NORM)
              .setNormFwdPhase(NormFwdPhase_t::TRAINING)
              .setxDesc(tensor(1, ACTIVATION))
              .setScaleAndBias(tensor(2, CHANNEL), tensor(3, CHANNEL))
              .setEpsilonTensor(scalar(4))
              .setyDesc(tensor(5, ACTIVATION))
              .setPrevRunningMeanAndVar(tensor(6, CHANNEL), tensor(7, CHANNEL))
              .setExpDecayFactorTensor(scalar(8))
              .setNextRunningMeanAndVar(tensor(9, CHANNEL), tensor(10, CHANNEL))
              .build();
    hipdnnBackendDescriptor_t hipdnnOp = lowered(op);
    ASSERT_NE(hipdnnOp, nullptr);
    EXPECT_EQ(typeOf(hipdnnOp), HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT);
    expectSlots(hipdnnOp,
                {{HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, 4},
                 {HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT, 6},
                 {HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT, 7},
                 {HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT, 8},
                 {HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT, 9},
                 {HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT, 10},
                 {HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, -1},
                 {HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT, -1}});
}

TEST_F(TestCudnnShimV8BuilderLowering, RmsNormHasNoMeanAndOptionalBias)
{
    v8fe::OperationBuilder withMean(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);
    withMean.setNormalizationMode(NormMode_t::RMS_NORM)
        .setNormFwdPhase(NormFwdPhase_t::TRAINING)
        .setxDesc(tensor(1, ACTIVATION))
        .setScale(tensor(2, LAYER_SCALE))
        .setEpsilonTensor(scalar(3))
        .setyDesc(tensor(4, ACTIVATION))
        .setSavedMeanAndInvVar(tensor(5, PER_SAMPLE), tensor(6, PER_SAMPLE));
    const hipdnn_shim_test::BuildFailure meanFailure = hipdnn_shim_test::failureOf([&] {
        const v8fe::Operation op = withMean.build();
        graph({&op});
    });
    ASSERT_TRUE(meanFailure.thrown);
    EXPECT_EQ(meanFailure.status, CUDNN_STATUS_NOT_SUPPORTED);

    const v8fe::Operation withBias = rmsNormBackward(true);
    hipdnnBackendDescriptor_t withBiasOp = lowered(withBias);
    ASSERT_NE(withBiasOp, nullptr);
    expectSlots(withBiasOp,
                {{HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT, 4},
                 {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT, 6},
                 {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT, 7}});

    const v8fe::Operation withoutBias = rmsNormBackward(false);
    hipdnnBackendDescriptor_t withoutBiasOp = lowered(withoutBias);
    ASSERT_NE(withoutBiasOp, nullptr);
    expectSlots(withoutBiasOp,
                {{HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT, 4},
                 {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT, 6},
                 {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT, -1}});
}

TEST_F(TestCudnnShimV8BuilderLowering, ResampleParametersAndIndexPresence)
{
    // 3x3 window, stride 2, padding differing per axis: IMAGE -> POOLED.
    const v8fe::ResampleDesc_v8 paddedMax = resample({v8fe::ResampleMode_t::MAXPOOL,
                                                      v8fe::PaddingMode_t::NEG_INF_PAD,
                                                      {3, 3},
                                                      {1, 0},
                                                      {0, 1},
                                                      {2, 2}});
    const v8fe::Tensor forwardIndex = indexTensor(3);
    const v8fe::Operation indexed
        = resampleForward(paddedMax, tensor(1, IMAGE), tensor(2, POOLED), &forwardIndex);
    hipdnnBackendDescriptor_t indexedOp = lowered(indexed);
    ASSERT_NE(indexedOp, nullptr);
    EXPECT_EQ(attributeOn(indexedOp, HIPDNN_ATTR_RESAMPLE_MODE).value<hipdnnResampleMode_t>(),
              HIPDNN_RESAMPLE_MAXPOOL);
    EXPECT_EQ(
        attributeOn(indexedOp, HIPDNN_ATTR_RESAMPLE_PADDING_MODE).value<hipdnnPaddingMode_t>(),
        HIPDNN_PADDING_NEG_INF_PAD);
    EXPECT_EQ(attributeOn(indexedOp, HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS).values<std::int64_t>(),
              (Dims{3, 3}));
    EXPECT_EQ(attributeOn(indexedOp, HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS).values<std::int64_t>(),
              (Dims{1, 0}));
    EXPECT_EQ(attributeOn(indexedOp, HIPDNN_ATTR_RESAMPLE_POST_PADDINGS).values<std::int64_t>(),
              (Dims{0, 1}));
    EXPECT_EQ(attributeOn(indexedOp, HIPDNN_ATTR_RESAMPLE_STRIDES).values<std::int64_t>(),
              (Dims{2, 2}));
    const auto generateIndex = callOn(indexedOp, HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT);
    ASSERT_TRUE(generateIndex);
    EXPECT_TRUE(generateIndex->value<bool>());
    expectSlots(indexedOp,
                {{HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, 1},
                 {HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, 2},
                 {HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC, 3}});

    const v8fe::Operation plain = resampleForward(maxPool(), tensor(4, IMAGE), tensor(5, POOLED));
    hipdnnBackendDescriptor_t plainOp = lowered(plain);
    ASSERT_NE(plainOp, nullptr);
    EXPECT_EQ(attributeOn(plainOp, HIPDNN_ATTR_RESAMPLE_PADDING_MODE).value<hipdnnPaddingMode_t>(),
              HIPDNN_PADDING_NEG_INF_PAD);
    EXPECT_FALSE(callOn(plainOp, HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT));
    expectSlots(plainOp,
                {{HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, 5},
                 {HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC, -1}});

    // Only max pooling produces indices; no provider takes one for another mode.
    for(const v8fe::ResampleMode_t mode : {v8fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING,
                                           v8fe::ResampleMode_t::AVGPOOL_EXCLUDE_PADDING})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        const v8fe::ResampleDesc_v8 average
            = resample({mode, v8fe::PaddingMode_t::ZERO_PAD, {2, 2}, {0, 0}, {0, 0}, {2, 2}});
        const v8fe::Tensor averageIndex = indexTensor(13);
        const std::vector<std::function<void()>> builds = {
            [&] {
                [[maybe_unused]] const v8fe::Operation op = resampleForward(
                    average, tensor(11, IMAGE), tensor(12, POOLED), &averageIndex);
            },
            [&] {
                [[maybe_unused]] const v8fe::Operation op = resampleBackward(
                    average, tensor(12, POOLED), tensor(11, IMAGE), &averageIndex);
            },
        };
        for(const std::function<void()>& build : builds)
        {
            const hipdnn_shim_test::BuildFailure failure = hipdnn_shim_test::failureOf(build);
            ASSERT_TRUE(failure.thrown);
            EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
            EXPECT_NE(failure.what.find("index tensor is not supported by the hipDNN cuDNN shim "
                                        "except for CUDNN_RESAMPLE_MAXPOOL"),
                      std::string::npos)
                << failure.what;
        }
    }

    // The backward op keeps the asymmetric forward geometry.
    const v8fe::Tensor backwardIndex = indexTensor(8);
    const v8fe::Operation maxBackward
        = resampleBackward(paddedMax, tensor(6, POOLED), tensor(7, IMAGE), &backwardIndex);
    hipdnnBackendDescriptor_t maxBackwardOp = lowered(maxBackward);
    ASSERT_NE(maxBackwardOp, nullptr);
    EXPECT_EQ(
        attributeOn(maxBackwardOp, HIPDNN_ATTR_RESAMPLE_PADDING_MODE).value<hipdnnPaddingMode_t>(),
        HIPDNN_PADDING_NEG_INF_PAD);
    EXPECT_EQ(attributeOn(maxBackwardOp, HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS).values<std::int64_t>(),
              (Dims{3, 3}));
    EXPECT_EQ(attributeOn(maxBackwardOp, HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS).values<std::int64_t>(),
              (Dims{1, 0}));
    EXPECT_EQ(attributeOn(maxBackwardOp, HIPDNN_ATTR_RESAMPLE_POST_PADDINGS).values<std::int64_t>(),
              (Dims{0, 1}));
    expectSlots(maxBackwardOp,
                {{HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, 6},
                 {HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, 7},
                 {HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC, 8}});

    const v8fe::Operation averageBackward
        = resampleBackward(averagePool(), tensor(9, POOLED), tensor(10, IMAGE));
    hipdnnBackendDescriptor_t averageBackwardOp = lowered(averageBackward);
    ASSERT_NE(averageBackwardOp, nullptr);
    EXPECT_EQ(attributeOn(averageBackwardOp, HIPDNN_ATTR_RESAMPLE_PADDING_MODE)
                  .value<hipdnnPaddingMode_t>(),
              HIPDNN_PADDING_ZERO_PAD);
    expectSlots(averageBackwardOp, {{HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC, -1}});
}

TEST_F(TestCudnnShimV8BuilderLowering, ReductionForwardsDeterminism)
{
    const v8fe::Operation unset = reduction(1, 2, std::nullopt);
    const v8fe::Operation deterministic = reduction(3, 4, true);
    graph({&unset, &deterministic});

    const std::vector<hipdnnBackendDescriptor_t> ops
        = created(HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
    ASSERT_EQ(ops.size(), 2U);
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_REDUCTION_YDESC), 2);
    EXPECT_EQ(attributeOn(ops[0], HIPDNN_ATTR_REDUCTION_OPERATOR).value<hipdnnReduceTensorOp_t>(),
              HIPDNN_REDUCE_TENSOR_ADD);
    EXPECT_FALSE(attributeOn(ops[0], HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC).value<bool>());
    EXPECT_EQ(slotUid(ops[1], HIPDNN_ATTR_OPERATION_REDUCTION_YDESC), 4);
    EXPECT_TRUE(attributeOn(ops[1], HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC).value<bool>());
}

} // namespace

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_utils.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//   include/cudnn_frontend/knobs.h
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

#pragma once

#include <optional>

#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>

// cuDNN C enum <-> shim FE enum, matched by enumerator name as upstream's
// convert_to_cudnn_type / convert_from_cudnn_type do. std::nullopt means the
// value is valid on one side but has no counterpart on the other; callers
// report NOT_SUPPORTED. Raw C values outside the declared range never reach
// these functions: SET rejects them with BAD_PARAM (enumValueCount).
// FE -> hipDNN C lowering reuses the Types.hpp converters (toHipdnnDataType,
// toBackendPointwiseMode, toBackendConvMode, ...).
namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

constexpr std::optional<DataType_t> toFe(cudnnDataType_t value) noexcept
{
    switch(value)
    {
    case CUDNN_DATA_BFLOAT16:
        return DataType_t::BFLOAT16;
    case CUDNN_DATA_BOOLEAN:
        return DataType_t::BOOLEAN;
    case CUDNN_DATA_COMPLEX_FP32:
        return DataType_t::COMPLEX_FP32;
    case CUDNN_DATA_COMPLEX_FP64:
        return DataType_t::COMPLEX_FP64;
    case CUDNN_DATA_DOUBLE:
        return DataType_t::DOUBLE;
    case CUDNN_DATA_FAST_FLOAT_FOR_FP8:
        return DataType_t::FAST_FLOAT_FOR_FP8;
    case CUDNN_DATA_FLOAT:
        return DataType_t::FLOAT;
    case CUDNN_DATA_FP4_E2M1:
        return DataType_t::FP4_E2M1;
    case CUDNN_DATA_FP8_E4M3:
        return DataType_t::FP8_E4M3;
    case CUDNN_DATA_FP8_E5M2:
        return DataType_t::FP8_E5M2;
    case CUDNN_DATA_FP8_E8M0:
        return DataType_t::FP8_E8M0;
    case CUDNN_DATA_HALF:
        return DataType_t::HALF;
    case CUDNN_DATA_INT32:
        return DataType_t::INT32;
    case CUDNN_DATA_INT4:
        return DataType_t::INT4;
    case CUDNN_DATA_INT64:
        return DataType_t::INT64;
    case CUDNN_DATA_INT8:
        return DataType_t::INT8;
    case CUDNN_DATA_INT8x32:
        return DataType_t::INT8x32;
    case CUDNN_DATA_INT8x4:
        return DataType_t::INT8x4;
    case CUDNN_DATA_UINT8:
        return DataType_t::UINT8;
    case CUDNN_DATA_UINT8x4:
        return DataType_t::UINT8x4;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnDataType_t> toCudnn(DataType_t value) noexcept
{
    switch(value)
    {
    case DataType_t::FLOAT:
        return CUDNN_DATA_FLOAT;
    case DataType_t::DOUBLE:
        return CUDNN_DATA_DOUBLE;
    case DataType_t::HALF:
        return CUDNN_DATA_HALF;
    case DataType_t::INT8:
        return CUDNN_DATA_INT8;
    case DataType_t::INT32:
        return CUDNN_DATA_INT32;
    case DataType_t::INT8x4:
        return CUDNN_DATA_INT8x4;
    case DataType_t::UINT8:
        return CUDNN_DATA_UINT8;
    case DataType_t::UINT8x4:
        return CUDNN_DATA_UINT8x4;
    case DataType_t::INT8x32:
        return CUDNN_DATA_INT8x32;
    case DataType_t::BFLOAT16:
        return CUDNN_DATA_BFLOAT16;
    case DataType_t::INT64:
        return CUDNN_DATA_INT64;
    case DataType_t::BOOLEAN:
        return CUDNN_DATA_BOOLEAN;
    case DataType_t::FP8_E4M3:
        return CUDNN_DATA_FP8_E4M3;
    case DataType_t::FP8_E5M2:
        return CUDNN_DATA_FP8_E5M2;
    case DataType_t::FAST_FLOAT_FOR_FP8:
        return CUDNN_DATA_FAST_FLOAT_FOR_FP8;
    case DataType_t::FP8_E8M0:
        return CUDNN_DATA_FP8_E8M0;
    case DataType_t::FP4_E2M1:
        return CUDNN_DATA_FP4_E2M1;
    case DataType_t::INT4:
        return CUDNN_DATA_INT4;
    case DataType_t::COMPLEX_FP32:
        return CUDNN_DATA_COMPLEX_FP32;
    case DataType_t::COMPLEX_FP64:
        return CUDNN_DATA_COMPLEX_FP64;
    case DataType_t::NOT_SET:
    // hipDNN-only types.
    case DataType_t::FP6_E2M3:
    case DataType_t::FP6_E3M2:
    case DataType_t::FP8_E4M3_FNUZ:
    case DataType_t::FP8_E5M2_FNUZ:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<PointwiseMode_t> toFe(cudnnPointwiseMode_t value) noexcept
{
    switch(value)
    {
    case CUDNN_POINTWISE_ABS:
        return PointwiseMode_t::ABS;
    case CUDNN_POINTWISE_ADD:
        return PointwiseMode_t::ADD;
    case CUDNN_POINTWISE_ADD_SQUARE:
        return PointwiseMode_t::ADD_SQUARE;
    case CUDNN_POINTWISE_BINARY_SELECT:
        return PointwiseMode_t::BINARY_SELECT;
    case CUDNN_POINTWISE_CEIL:
        return PointwiseMode_t::CEIL;
    case CUDNN_POINTWISE_CMP_EQ:
        return PointwiseMode_t::CMP_EQ;
    case CUDNN_POINTWISE_CMP_GE:
        return PointwiseMode_t::CMP_GE;
    case CUDNN_POINTWISE_CMP_GT:
        return PointwiseMode_t::CMP_GT;
    case CUDNN_POINTWISE_CMP_LE:
        return PointwiseMode_t::CMP_LE;
    case CUDNN_POINTWISE_CMP_LT:
        return PointwiseMode_t::CMP_LT;
    case CUDNN_POINTWISE_CMP_NEQ:
        return PointwiseMode_t::CMP_NEQ;
    case CUDNN_POINTWISE_COS:
        return PointwiseMode_t::COS;
    case CUDNN_POINTWISE_DIV:
        return PointwiseMode_t::DIV;
    case CUDNN_POINTWISE_ELU_BWD:
        return PointwiseMode_t::ELU_BWD;
    case CUDNN_POINTWISE_ELU_FWD:
        return PointwiseMode_t::ELU_FWD;
    case CUDNN_POINTWISE_ERF:
        return PointwiseMode_t::ERF;
    case CUDNN_POINTWISE_EXP:
        return PointwiseMode_t::EXP;
    case CUDNN_POINTWISE_FLOOR:
        return PointwiseMode_t::FLOOR;
    case CUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
        return PointwiseMode_t::GELU_APPROX_TANH_BWD;
    case CUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
        return PointwiseMode_t::GELU_APPROX_TANH_FWD;
    case CUDNN_POINTWISE_GELU_BWD:
        return PointwiseMode_t::GELU_BWD;
    case CUDNN_POINTWISE_GELU_FWD:
        return PointwiseMode_t::GELU_FWD;
    case CUDNN_POINTWISE_GEN_INDEX:
        return PointwiseMode_t::GEN_INDEX;
    case CUDNN_POINTWISE_IDENTITY:
        return PointwiseMode_t::IDENTITY;
    case CUDNN_POINTWISE_LOG:
        return PointwiseMode_t::LOG;
    case CUDNN_POINTWISE_LOGICAL_AND:
        return PointwiseMode_t::LOGICAL_AND;
    case CUDNN_POINTWISE_LOGICAL_NOT:
        return PointwiseMode_t::LOGICAL_NOT;
    case CUDNN_POINTWISE_LOGICAL_OR:
        return PointwiseMode_t::LOGICAL_OR;
    case CUDNN_POINTWISE_MAX:
        return PointwiseMode_t::MAX;
    case CUDNN_POINTWISE_MIN:
        return PointwiseMode_t::MIN;
    case CUDNN_POINTWISE_MOD:
        return PointwiseMode_t::MOD;
    case CUDNN_POINTWISE_MUL:
        return PointwiseMode_t::MUL;
    case CUDNN_POINTWISE_NEG:
        return PointwiseMode_t::NEG;
    case CUDNN_POINTWISE_POW:
        return PointwiseMode_t::POW;
    case CUDNN_POINTWISE_RECIPROCAL:
        return PointwiseMode_t::RECIPROCAL;
    case CUDNN_POINTWISE_RELU_BWD:
        return PointwiseMode_t::RELU_BWD;
    case CUDNN_POINTWISE_RELU_FWD:
        return PointwiseMode_t::RELU_FWD;
    case CUDNN_POINTWISE_RSQRT:
        return PointwiseMode_t::RSQRT;
    case CUDNN_POINTWISE_SIGMOID_BWD:
        return PointwiseMode_t::SIGMOID_BWD;
    case CUDNN_POINTWISE_SIGMOID_FWD:
        return PointwiseMode_t::SIGMOID_FWD;
    case CUDNN_POINTWISE_SIN:
        return PointwiseMode_t::SIN;
    case CUDNN_POINTWISE_SOFTPLUS_BWD:
        return PointwiseMode_t::SOFTPLUS_BWD;
    case CUDNN_POINTWISE_SOFTPLUS_FWD:
        return PointwiseMode_t::SOFTPLUS_FWD;
    case CUDNN_POINTWISE_SQRT:
        return PointwiseMode_t::SQRT;
    case CUDNN_POINTWISE_SUB:
        return PointwiseMode_t::SUB;
    case CUDNN_POINTWISE_SWISH_BWD:
        return PointwiseMode_t::SWISH_BWD;
    case CUDNN_POINTWISE_SWISH_FWD:
        return PointwiseMode_t::SWISH_FWD;
    case CUDNN_POINTWISE_TAN:
        return PointwiseMode_t::TAN;
    case CUDNN_POINTWISE_TANH_BWD:
        return PointwiseMode_t::TANH_BWD;
    case CUDNN_POINTWISE_TANH_FWD:
        return PointwiseMode_t::TANH_FWD;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnPointwiseMode_t> toCudnn(PointwiseMode_t value) noexcept
{
    switch(value)
    {
    case PointwiseMode_t::ADD:
        return CUDNN_POINTWISE_ADD;
    case PointwiseMode_t::MUL:
        return CUDNN_POINTWISE_MUL;
    case PointwiseMode_t::SQRT:
        return CUDNN_POINTWISE_SQRT;
    case PointwiseMode_t::MAX:
        return CUDNN_POINTWISE_MAX;
    case PointwiseMode_t::MIN:
        return CUDNN_POINTWISE_MIN;
    case PointwiseMode_t::RELU_FWD:
        return CUDNN_POINTWISE_RELU_FWD;
    case PointwiseMode_t::TANH_FWD:
        return CUDNN_POINTWISE_TANH_FWD;
    case PointwiseMode_t::SIGMOID_FWD:
        return CUDNN_POINTWISE_SIGMOID_FWD;
    case PointwiseMode_t::ELU_FWD:
        return CUDNN_POINTWISE_ELU_FWD;
    case PointwiseMode_t::GELU_FWD:
        return CUDNN_POINTWISE_GELU_FWD;
    case PointwiseMode_t::SOFTPLUS_FWD:
        return CUDNN_POINTWISE_SOFTPLUS_FWD;
    case PointwiseMode_t::SWISH_FWD:
        return CUDNN_POINTWISE_SWISH_FWD;
    case PointwiseMode_t::RELU_BWD:
        return CUDNN_POINTWISE_RELU_BWD;
    case PointwiseMode_t::TANH_BWD:
        return CUDNN_POINTWISE_TANH_BWD;
    case PointwiseMode_t::SIGMOID_BWD:
        return CUDNN_POINTWISE_SIGMOID_BWD;
    case PointwiseMode_t::ELU_BWD:
        return CUDNN_POINTWISE_ELU_BWD;
    case PointwiseMode_t::GELU_BWD:
        return CUDNN_POINTWISE_GELU_BWD;
    case PointwiseMode_t::SOFTPLUS_BWD:
        return CUDNN_POINTWISE_SOFTPLUS_BWD;
    case PointwiseMode_t::SWISH_BWD:
        return CUDNN_POINTWISE_SWISH_BWD;
    case PointwiseMode_t::ERF:
        return CUDNN_POINTWISE_ERF;
    case PointwiseMode_t::IDENTITY:
        return CUDNN_POINTWISE_IDENTITY;
    case PointwiseMode_t::GELU_APPROX_TANH_BWD:
        return CUDNN_POINTWISE_GELU_APPROX_TANH_BWD;
    case PointwiseMode_t::GELU_APPROX_TANH_FWD:
        return CUDNN_POINTWISE_GELU_APPROX_TANH_FWD;
    case PointwiseMode_t::GEN_INDEX:
        return CUDNN_POINTWISE_GEN_INDEX;
    case PointwiseMode_t::BINARY_SELECT:
        return CUDNN_POINTWISE_BINARY_SELECT;
    case PointwiseMode_t::EXP:
        return CUDNN_POINTWISE_EXP;
    case PointwiseMode_t::LOG:
        return CUDNN_POINTWISE_LOG;
    case PointwiseMode_t::NEG:
        return CUDNN_POINTWISE_NEG;
    case PointwiseMode_t::MOD:
        return CUDNN_POINTWISE_MOD;
    case PointwiseMode_t::POW:
        return CUDNN_POINTWISE_POW;
    case PointwiseMode_t::ABS:
        return CUDNN_POINTWISE_ABS;
    case PointwiseMode_t::CEIL:
        return CUDNN_POINTWISE_CEIL;
    case PointwiseMode_t::COS:
        return CUDNN_POINTWISE_COS;
    case PointwiseMode_t::FLOOR:
        return CUDNN_POINTWISE_FLOOR;
    case PointwiseMode_t::RSQRT:
        return CUDNN_POINTWISE_RSQRT;
    case PointwiseMode_t::SIN:
        return CUDNN_POINTWISE_SIN;
    case PointwiseMode_t::LOGICAL_NOT:
        return CUDNN_POINTWISE_LOGICAL_NOT;
    case PointwiseMode_t::TAN:
        return CUDNN_POINTWISE_TAN;
    case PointwiseMode_t::SUB:
        return CUDNN_POINTWISE_SUB;
    case PointwiseMode_t::ADD_SQUARE:
        return CUDNN_POINTWISE_ADD_SQUARE;
    case PointwiseMode_t::DIV:
        return CUDNN_POINTWISE_DIV;
    case PointwiseMode_t::CMP_EQ:
        return CUDNN_POINTWISE_CMP_EQ;
    case PointwiseMode_t::CMP_NEQ:
        return CUDNN_POINTWISE_CMP_NEQ;
    case PointwiseMode_t::CMP_GT:
        return CUDNN_POINTWISE_CMP_GT;
    case PointwiseMode_t::CMP_GE:
        return CUDNN_POINTWISE_CMP_GE;
    case PointwiseMode_t::CMP_LT:
        return CUDNN_POINTWISE_CMP_LT;
    case PointwiseMode_t::CMP_LE:
        return CUDNN_POINTWISE_CMP_LE;
    case PointwiseMode_t::LOGICAL_AND:
        return CUDNN_POINTWISE_LOGICAL_AND;
    case PointwiseMode_t::LOGICAL_OR:
        return CUDNN_POINTWISE_LOGICAL_OR;
    case PointwiseMode_t::RECIPROCAL:
        return CUDNN_POINTWISE_RECIPROCAL;
    case PointwiseMode_t::NOT_SET:
    // Sentinel, not a mode.
    case PointwiseMode_t::COUNT:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<ConvolutionMode_t> toFe(cudnnConvolutionMode_t value) noexcept
{
    switch(value)
    {
    case CUDNN_CONVOLUTION:
        return ConvolutionMode_t::CONVOLUTION;
    case CUDNN_CROSS_CORRELATION:
        return ConvolutionMode_t::CROSS_CORRELATION;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnConvolutionMode_t> toCudnn(ConvolutionMode_t value) noexcept
{
    switch(value)
    {
    case ConvolutionMode_t::CONVOLUTION:
        return CUDNN_CONVOLUTION;
    case ConvolutionMode_t::CROSS_CORRELATION:
        return CUDNN_CROSS_CORRELATION;
    case ConvolutionMode_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<ReductionMode_t> toFe(cudnnReduceTensorOp_t value) noexcept
{
    switch(value)
    {
    case CUDNN_REDUCE_TENSOR_ADD:
        return ReductionMode_t::ADD;
    case CUDNN_REDUCE_TENSOR_AMAX:
        return ReductionMode_t::AMAX;
    case CUDNN_REDUCE_TENSOR_AVG:
        return ReductionMode_t::AVG;
    case CUDNN_REDUCE_TENSOR_MAX:
        return ReductionMode_t::MAX;
    case CUDNN_REDUCE_TENSOR_MIN:
        return ReductionMode_t::MIN;
    case CUDNN_REDUCE_TENSOR_MUL:
        return ReductionMode_t::MUL;
    case CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
        return ReductionMode_t::MUL_NO_ZEROS;
    case CUDNN_REDUCE_TENSOR_NORM1:
        return ReductionMode_t::NORM1;
    case CUDNN_REDUCE_TENSOR_NORM2:
        return ReductionMode_t::NORM2;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnReduceTensorOp_t> toCudnn(ReductionMode_t value) noexcept
{
    switch(value)
    {
    case ReductionMode_t::ADD:
        return CUDNN_REDUCE_TENSOR_ADD;
    case ReductionMode_t::MUL:
        return CUDNN_REDUCE_TENSOR_MUL;
    case ReductionMode_t::MIN:
        return CUDNN_REDUCE_TENSOR_MIN;
    case ReductionMode_t::MAX:
        return CUDNN_REDUCE_TENSOR_MAX;
    case ReductionMode_t::AMAX:
        return CUDNN_REDUCE_TENSOR_AMAX;
    case ReductionMode_t::AVG:
        return CUDNN_REDUCE_TENSOR_AVG;
    case ReductionMode_t::NORM1:
        return CUDNN_REDUCE_TENSOR_NORM1;
    case ReductionMode_t::NORM2:
        return CUDNN_REDUCE_TENSOR_NORM2;
    case ReductionMode_t::MUL_NO_ZEROS:
        return CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS;
    case ReductionMode_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<ResampleMode_t> toFe(cudnnResampleMode_t value) noexcept
{
    switch(value)
    {
    // cuDNN defines AVGPOOL as an alias of AVGPOOL_INCLUDE_PADDING.
    case CUDNN_RESAMPLE_AVGPOOL:
        return ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
    case CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING:
        return ResampleMode_t::AVGPOOL_EXCLUDE_PADDING;
    case CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING:
        return ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
    case CUDNN_RESAMPLE_BILINEAR:
        return ResampleMode_t::BILINEAR;
    case CUDNN_RESAMPLE_MAXPOOL:
        return ResampleMode_t::MAXPOOL;
    case CUDNN_RESAMPLE_NEAREST:
        return ResampleMode_t::NEAREST;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnResampleMode_t> toCudnn(ResampleMode_t value) noexcept
{
    switch(value)
    {
    case ResampleMode_t::AVGPOOL_EXCLUDE_PADDING:
        return CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;
    case ResampleMode_t::AVGPOOL_INCLUDE_PADDING:
        return CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
    case ResampleMode_t::BILINEAR:
        return CUDNN_RESAMPLE_BILINEAR;
    case ResampleMode_t::NEAREST:
        return CUDNN_RESAMPLE_NEAREST;
    case ResampleMode_t::MAXPOOL:
        return CUDNN_RESAMPLE_MAXPOOL;
    case ResampleMode_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<PaddingMode_t> toFe(cudnnPaddingMode_t value) noexcept
{
    switch(value)
    {
    case CUDNN_EDGE_VAL_PAD:
        return PaddingMode_t::EDGE_VAL_PAD;
    case CUDNN_NEG_INF_PAD:
        return PaddingMode_t::NEG_INF_PAD;
    case CUDNN_ZERO_PAD:
        return PaddingMode_t::ZERO_PAD;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnPaddingMode_t> toCudnn(PaddingMode_t value) noexcept
{
    switch(value)
    {
    case PaddingMode_t::EDGE_VAL_PAD:
        return CUDNN_EDGE_VAL_PAD;
    case PaddingMode_t::NEG_INF_PAD:
        return CUDNN_NEG_INF_PAD;
    case PaddingMode_t::ZERO_PAD:
        return CUDNN_ZERO_PAD;
    case PaddingMode_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<NormFwdPhase_t> toFe(cudnnBackendNormFwdPhase_t value) noexcept
{
    switch(value)
    {
    case CUDNN_NORM_FWD_INFERENCE:
        return NormFwdPhase_t::INFERENCE;
    case CUDNN_NORM_FWD_TRAINING:
        return NormFwdPhase_t::TRAINING;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnBackendNormFwdPhase_t> toCudnn(NormFwdPhase_t value) noexcept
{
    switch(value)
    {
    case NormFwdPhase_t::INFERENCE:
        return CUDNN_NORM_FWD_INFERENCE;
    case NormFwdPhase_t::TRAINING:
        return CUDNN_NORM_FWD_TRAINING;
    case NormFwdPhase_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<NormMode_t> toFe(cudnnBackendNormMode_t value) noexcept
{
    switch(value)
    {
    case CUDNN_ADA_LAYER_NORM:
        return NormMode_t::ADA_LAYER_NORM;
    case CUDNN_BATCH_NORM:
        return NormMode_t::BATCH_NORM;
    case CUDNN_GROUP_NORM:
        return NormMode_t::GROUP_NORM;
    case CUDNN_INSTANCE_NORM:
        return NormMode_t::INSTANCE_NORM;
    case CUDNN_LAYER_NORM:
        return NormMode_t::LAYER_NORM;
    case CUDNN_RMS_NORM:
        return NormMode_t::RMS_NORM;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnBackendNormMode_t> toCudnn(NormMode_t value) noexcept
{
    switch(value)
    {
    case NormMode_t::LAYER_NORM:
        return CUDNN_LAYER_NORM;
    case NormMode_t::INSTANCE_NORM:
        return CUDNN_INSTANCE_NORM;
    case NormMode_t::BATCH_NORM:
        return CUDNN_BATCH_NORM;
    case NormMode_t::GROUP_NORM:
        return CUDNN_GROUP_NORM;
    case NormMode_t::RMS_NORM:
        return CUDNN_RMS_NORM;
    case NormMode_t::ADA_LAYER_NORM:
        return CUDNN_ADA_LAYER_NORM;
    case NormMode_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<BehaviorNote_t> toFe(cudnnBackendBehaviorNote_t value) noexcept
{
    switch(value)
    {
    case CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY:
        return BehaviorNote_t::CUBLASLT_DEPENDENCY;
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
        return BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER;
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
        return BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER;
    case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
        return BehaviorNote_t::RUNTIME_COMPILATION;
    case CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API:
        return BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnBackendBehaviorNote_t> toCudnn(BehaviorNote_t value) noexcept
{
    switch(value)
    {
    case BehaviorNote_t::RUNTIME_COMPILATION:
        return CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
    case BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER:
        return CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER;
    case BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER:
        return CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER;
    case BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API:
        return CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API;
    case BehaviorNote_t::CUBLASLT_DEPENDENCY:
        return CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY;
    case BehaviorNote_t::NOT_SET:
    // hipDNN-only notes.
    case BehaviorNote_t::REQUIRES_LAYOUT_TRANSFORM:
    case BehaviorNote_t::SUPPORTS_GRAPH_CAPTURE:
    case BehaviorNote_t::EXTERNAL_LIBRARY_DEPENDENCY:
    case BehaviorNote_t::SUPPORTS_EXECUTION_PLAN_SERIALIZATION:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<NumericalNote_t> toFe(cudnnBackendNumericalNote_t value) noexcept
{
    switch(value)
    {
    case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
        return NumericalNote_t::DOWN_CONVERT_INPUTS;
    case CUDNN_NUMERICAL_NOTE_FFT:
        return NumericalNote_t::FFT;
    case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
        return NumericalNote_t::NONDETERMINISTIC;
    case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
        return NumericalNote_t::REDUCED_PRECISION_REDUCTION;
    case CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP:
        return NumericalNote_t::STRICT_NAN_PROP;
    case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
        return NumericalNote_t::TENSOR_CORE;
    case CUDNN_NUMERICAL_NOTE_WINOGRAD:
        return NumericalNote_t::WINOGRAD;
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13:
        return NumericalNote_t::WINOGRAD_TILE_13x13;
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
        return NumericalNote_t::WINOGRAD_TILE_4x4;
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
        return NumericalNote_t::WINOGRAD_TILE_6x6;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnBackendNumericalNote_t> toCudnn(NumericalNote_t value) noexcept
{
    switch(value)
    {
    case NumericalNote_t::TENSOR_CORE:
        return CUDNN_NUMERICAL_NOTE_TENSOR_CORE;
    case NumericalNote_t::DOWN_CONVERT_INPUTS:
        return CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS;
    case NumericalNote_t::REDUCED_PRECISION_REDUCTION:
        return CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION;
    case NumericalNote_t::FFT:
        return CUDNN_NUMERICAL_NOTE_FFT;
    case NumericalNote_t::NONDETERMINISTIC:
        return CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC;
    case NumericalNote_t::WINOGRAD:
        return CUDNN_NUMERICAL_NOTE_WINOGRAD;
    case NumericalNote_t::WINOGRAD_TILE_4x4:
        return CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4;
    case NumericalNote_t::WINOGRAD_TILE_6x6:
        return CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6;
    case NumericalNote_t::WINOGRAD_TILE_13x13:
        return CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13;
    case NumericalNote_t::STRICT_NAN_PROP:
        return CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP;
    case NumericalNote_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<TensorReordering_t> toFe(cudnnBackendTensorReordering_t value) noexcept
{
    switch(value)
    {
    case CUDNN_TENSOR_REORDERING_NONE:
        return TensorReordering_t::NONE;
    case CUDNN_TENSOR_REORDERING_INT8x32:
        return TensorReordering_t::INT8x32;
    case CUDNN_TENSOR_REORDERING_F16x16:
        return TensorReordering_t::F16x16;
    case CUDNN_TENSOR_REORDERING_F8_128x4:
        return TensorReordering_t::F8_128x4;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnBackendTensorReordering_t> toCudnn(TensorReordering_t value) noexcept
{
    switch(value)
    {
    case TensorReordering_t::NONE:
        return CUDNN_TENSOR_REORDERING_NONE;
    case TensorReordering_t::INT8x32:
        return CUDNN_TENSOR_REORDERING_INT8x32;
    case TensorReordering_t::F16x16:
        return CUDNN_TENSOR_REORDERING_F16x16;
    case TensorReordering_t::F8_128x4:
        return CUDNN_TENSOR_REORDERING_F8_128x4;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<ReshapeMode_t> toFe(cudnnBackendReshapeMode_t value) noexcept
{
    switch(value)
    {
    case CUDNN_RESHAPE_LOGICAL:
        return ReshapeMode_t::LOGICAL;
    case CUDNN_RESHAPE_VIEW_ONLY:
        return ReshapeMode_t::VIEW_ONLY;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnBackendReshapeMode_t> toCudnn(ReshapeMode_t value) noexcept
{
    switch(value)
    {
    case ReshapeMode_t::VIEW_ONLY:
        return CUDNN_RESHAPE_VIEW_ONLY;
    case ReshapeMode_t::LOGICAL:
        return CUDNN_RESHAPE_LOGICAL;
    case ReshapeMode_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<KnobType_t> toFe(cudnnBackendKnobType_t value) noexcept
{
    switch(value)
    {
    case CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD:
        return KnobType_t::ARRAY_SIZE_PER_THREAD;
    case CUDNN_KNOB_TYPE_BLOCK_SIZE:
        return KnobType_t::BLOCK_SIZE;
    case CUDNN_KNOB_TYPE_CTA_COUNT:
        return KnobType_t::CTA_COUNT;
    case CUDNN_KNOB_TYPE_EDGE:
        return KnobType_t::EDGE;
    case CUDNN_KNOB_TYPE_IDX_MODE:
        return KnobType_t::IDX_MODE;
    case CUDNN_KNOB_TYPE_KERNEL_CFG:
        return KnobType_t::KERNEL_CFG;
    case CUDNN_KNOB_TYPE_LOAD_SIZE:
        return KnobType_t::LOAD_SIZE;
    case CUDNN_KNOB_TYPE_MULTIPLY:
        return KnobType_t::MULTIPLY;
    case CUDNN_KNOB_TYPE_OCCUPANCY:
        return KnobType_t::OCCUPANCY;
    case CUDNN_KNOB_TYPE_REDUCTION_MODE:
        return KnobType_t::REDUCTION_MODE;
    case CUDNN_KNOB_TYPE_SPECFILT:
        return KnobType_t::SPECFILT;
    case CUDNN_KNOB_TYPE_SPLIT_COLS:
        return KnobType_t::SPLIT_COLS;
    case CUDNN_KNOB_TYPE_SPLIT_K_BUF:
        return KnobType_t::SPLIT_K_BUF;
    case CUDNN_KNOB_TYPE_SPLIT_K_SLC:
        return KnobType_t::SPLIT_K_SLC;
    case CUDNN_KNOB_TYPE_SPLIT_P_SLC:
        return KnobType_t::SPLIT_P_SLC;
    case CUDNN_KNOB_TYPE_STAGES:
        return KnobType_t::STAGES;
    case CUDNN_KNOB_TYPE_STREAM_K:
        return KnobType_t::STREAM_K;
    case CUDNN_KNOB_TYPE_SWIZZLE:
        return KnobType_t::SWIZZLE;
    case CUDNN_KNOB_TYPE_TILEK:
        return KnobType_t::TILEK;
    case CUDNN_KNOB_TYPE_TILE_CGA_M:
        return KnobType_t::TILE_CGA_M;
    case CUDNN_KNOB_TYPE_TILE_CGA_N:
        return KnobType_t::TILE_CGA_N;
    case CUDNN_KNOB_TYPE_TILE_COLS:
        return KnobType_t::TILE_COLS;
    case CUDNN_KNOB_TYPE_TILE_M:
        return KnobType_t::TILE_M;
    case CUDNN_KNOB_TYPE_TILE_N:
        return KnobType_t::TILE_N;
    case CUDNN_KNOB_TYPE_TILE_ROWS:
        return KnobType_t::TILE_ROWS;
    case CUDNN_KNOB_TYPE_TILE_SIZE:
        return KnobType_t::TILE_SIZE;
    case CUDNN_KNOB_TYPE_WARP_SPEC_CFG:
        return KnobType_t::WARP_SPEC_CFG;
    case CUDNN_KNOB_TYPE_WORKSPACE:
        return KnobType_t::WORKSPACE;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnBackendKnobType_t> toCudnn(KnobType_t value) noexcept
{
    switch(value)
    {
    case KnobType_t::SWIZZLE:
        return CUDNN_KNOB_TYPE_SWIZZLE;
    case KnobType_t::TILE_SIZE:
        return CUDNN_KNOB_TYPE_TILE_SIZE;
    case KnobType_t::EDGE:
        return CUDNN_KNOB_TYPE_EDGE;
    case KnobType_t::MULTIPLY:
        return CUDNN_KNOB_TYPE_MULTIPLY;
    case KnobType_t::SPLIT_K_BUF:
        return CUDNN_KNOB_TYPE_SPLIT_K_BUF;
    case KnobType_t::TILEK:
        return CUDNN_KNOB_TYPE_TILEK;
    case KnobType_t::STAGES:
        return CUDNN_KNOB_TYPE_STAGES;
    case KnobType_t::REDUCTION_MODE:
        return CUDNN_KNOB_TYPE_REDUCTION_MODE;
    case KnobType_t::SPLIT_K_SLC:
        return CUDNN_KNOB_TYPE_SPLIT_K_SLC;
    case KnobType_t::IDX_MODE:
        return CUDNN_KNOB_TYPE_IDX_MODE;
    case KnobType_t::SPECFILT:
        return CUDNN_KNOB_TYPE_SPECFILT;
    case KnobType_t::KERNEL_CFG:
        return CUDNN_KNOB_TYPE_KERNEL_CFG;
    case KnobType_t::WORKSPACE:
        return CUDNN_KNOB_TYPE_WORKSPACE;
    case KnobType_t::TILE_CGA_M:
        return CUDNN_KNOB_TYPE_TILE_CGA_M;
    case KnobType_t::TILE_CGA_N:
        return CUDNN_KNOB_TYPE_TILE_CGA_N;
    case KnobType_t::BLOCK_SIZE:
        return CUDNN_KNOB_TYPE_BLOCK_SIZE;
    case KnobType_t::OCCUPANCY:
        return CUDNN_KNOB_TYPE_OCCUPANCY;
    case KnobType_t::ARRAY_SIZE_PER_THREAD:
        return CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD;
    case KnobType_t::SPLIT_COLS:
        return CUDNN_KNOB_TYPE_SPLIT_COLS;
    case KnobType_t::TILE_ROWS:
        return CUDNN_KNOB_TYPE_TILE_ROWS;
    case KnobType_t::TILE_COLS:
        return CUDNN_KNOB_TYPE_TILE_COLS;
    case KnobType_t::LOAD_SIZE:
        return CUDNN_KNOB_TYPE_LOAD_SIZE;
    case KnobType_t::CTA_COUNT:
        return CUDNN_KNOB_TYPE_CTA_COUNT;
    case KnobType_t::STREAM_K:
        return CUDNN_KNOB_TYPE_STREAM_K;
    case KnobType_t::SPLIT_P_SLC:
        return CUDNN_KNOB_TYPE_SPLIT_P_SLC;
    case KnobType_t::TILE_M:
        return CUDNN_KNOB_TYPE_TILE_M;
    case KnobType_t::TILE_N:
        return CUDNN_KNOB_TYPE_TILE_N;
    case KnobType_t::WARP_SPEC_CFG:
        return CUDNN_KNOB_TYPE_WARP_SPEC_CFG;
    case KnobType_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<HeurMode_t> toFe(cudnnBackendHeurMode_t value) noexcept
{
    switch(value)
    {
    case CUDNN_HEUR_MODE_A:
        return HeurMode_t::A;
    case CUDNN_HEUR_MODE_B:
        return HeurMode_t::B;
    case CUDNN_HEUR_MODE_FALLBACK:
        return HeurMode_t::FALLBACK;
    // The FE enum has no INSTANT mode.
    case CUDNN_HEUR_MODE_INSTANT:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnBackendHeurMode_t> toCudnn(HeurMode_t value) noexcept
{
    switch(value)
    {
    case HeurMode_t::A:
        return CUDNN_HEUR_MODE_A;
    case HeurMode_t::B:
        return CUDNN_HEUR_MODE_B;
    case HeurMode_t::FALLBACK:
        return CUDNN_HEUR_MODE_FALLBACK;
    // Selects open-source heuristics; there is no backend mode for it.
    case HeurMode_t::OPENSOURCE:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<RngDistribution_t> toFe(cudnnRngDistribution_t value) noexcept
{
    switch(value)
    {
    case CUDNN_RNG_DISTRIBUTION_BERNOULLI:
        return RngDistribution_t::BERNOULLI;
    case CUDNN_RNG_DISTRIBUTION_NORMAL:
        return RngDistribution_t::NORMAL;
    case CUDNN_RNG_DISTRIBUTION_UNIFORM:
        return RngDistribution_t::UNIFORM;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnRngDistribution_t> toCudnn(RngDistribution_t value) noexcept
{
    switch(value)
    {
    case RngDistribution_t::BERNOULLI:
        return CUDNN_RNG_DISTRIBUTION_BERNOULLI;
    case RngDistribution_t::UNIFORM:
        return CUDNN_RNG_DISTRIBUTION_UNIFORM;
    case RngDistribution_t::NORMAL:
        return CUDNN_RNG_DISTRIBUTION_NORMAL;
    case RngDistribution_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<MoeGroupedMatmulMode_t> toFe(cudnnMoeGroupedMatmulMode_t value) noexcept
{
    switch(value)
    {
    case CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER:
        return MoeGroupedMatmulMode_t::GATHER;
    case CUDNN_MOE_GROUPED_MATMUL_MODE_NONE:
        return MoeGroupedMatmulMode_t::NONE;
    case CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER:
        return MoeGroupedMatmulMode_t::SCATTER;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnMoeGroupedMatmulMode_t> toCudnn(MoeGroupedMatmulMode_t value) noexcept
{
    switch(value)
    {
    case MoeGroupedMatmulMode_t::NONE:
        return CUDNN_MOE_GROUPED_MATMUL_MODE_NONE;
    case MoeGroupedMatmulMode_t::GATHER:
        return CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER;
    case MoeGroupedMatmulMode_t::SCATTER:
        return CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;
    case MoeGroupedMatmulMode_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

constexpr std::optional<DescriptorType_t> toFe(cudnnBackendDescriptorType_t value) noexcept
{
    switch(value)
    {
    case CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
        return DescriptorType_t::CONVOLUTION_DESCRIPTOR;
    case CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
        return DescriptorType_t::ENGINECFG_DESCRIPTOR;
    case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
        return DescriptorType_t::ENGINEHEUR_DESCRIPTOR;
    case CUDNN_BACKEND_ENGINE_DESCRIPTOR:
        return DescriptorType_t::ENGINE_DESCRIPTOR;
    case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
        return DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR;
    case CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
        return DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR;
    case CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
        return DescriptorType_t::KNOB_CHOICE_DESCRIPTOR;
    case CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
        return DescriptorType_t::KNOB_INFO_DESCRIPTOR;
    case CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR:
        return DescriptorType_t::LAYOUT_INFO_DESCRIPTOR;
    case CUDNN_BACKEND_MATMUL_DESCRIPTOR:
        return DescriptorType_t::MATMUL_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
        return DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
        return DescriptorType_t::OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
        return DescriptorType_t::OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
        return DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
        return DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
        return DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
        return DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
        return DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
        return DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
        return DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
        return DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR:
        return DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
        return DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
        return DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
        return DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
        return DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
        return DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
        return DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
        return DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
        return DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
        return DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
        return DescriptorType_t::OPERATION_RNG_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR:
        return DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR:
        return DescriptorType_t::OPERATION_SLICE_DESCRIPTOR;
    case CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR:
        return DescriptorType_t::OPERATION_TRANSPOSE_DESCRIPTOR;
    case CUDNN_BACKEND_POINTWISE_DESCRIPTOR:
        return DescriptorType_t::POINTWISE_DESCRIPTOR;
    case CUDNN_BACKEND_REDUCTION_DESCRIPTOR:
        return DescriptorType_t::REDUCTION_DESCRIPTOR;
    case CUDNN_BACKEND_RESAMPLE_DESCRIPTOR:
        return DescriptorType_t::RESAMPLE_DESCRIPTOR;
    case CUDNN_BACKEND_RNG_DESCRIPTOR:
        return DescriptorType_t::RNG_DESCRIPTOR;
    case CUDNN_BACKEND_TENSOR_DESCRIPTOR:
        return DescriptorType_t::TENSOR_DESCRIPTOR;
    case CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
        return DescriptorType_t::VARIANT_PACK_DESCRIPTOR;
    default:
        return std::nullopt;
    }
}

constexpr std::optional<cudnnBackendDescriptorType_t> toCudnn(DescriptorType_t value) noexcept
{
    switch(value)
    {
    case DescriptorType_t::POINTWISE_DESCRIPTOR:
        return CUDNN_BACKEND_POINTWISE_DESCRIPTOR;
    case DescriptorType_t::CONVOLUTION_DESCRIPTOR:
        return CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR;
    case DescriptorType_t::ENGINE_DESCRIPTOR:
        return CUDNN_BACKEND_ENGINE_DESCRIPTOR;
    case DescriptorType_t::ENGINECFG_DESCRIPTOR:
        return CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
    case DescriptorType_t::ENGINEHEUR_DESCRIPTOR:
        return CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
    case DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR:
        return CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
    case DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR:
        return CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR;
    case DescriptorType_t::KNOB_CHOICE_DESCRIPTOR:
        return CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR;
    case DescriptorType_t::KNOB_INFO_DESCRIPTOR:
        return CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR;
    case DescriptorType_t::LAYOUT_INFO_DESCRIPTOR:
        return CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR;
    case DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
    case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
    case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
    case DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
    case DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR;
    case DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
    case DescriptorType_t::VARIANT_PACK_DESCRIPTOR:
        return CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
    case DescriptorType_t::TENSOR_DESCRIPTOR:
        return CUDNN_BACKEND_TENSOR_DESCRIPTOR;
    case DescriptorType_t::MATMUL_DESCRIPTOR:
        return CUDNN_BACKEND_MATMUL_DESCRIPTOR;
    case DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR;
    case DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR;
    case DescriptorType_t::REDUCTION_DESCRIPTOR:
        return CUDNN_BACKEND_REDUCTION_DESCRIPTOR;
    case DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR;
    case DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR;
    case DescriptorType_t::RESAMPLE_DESCRIPTOR:
        return CUDNN_BACKEND_RESAMPLE_DESCRIPTOR;
    case DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR;
    case DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR;
    case DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR;
    case DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR;
    case DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
    case DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR;
    case DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR;
    case DescriptorType_t::RNG_DESCRIPTOR:
        return CUDNN_BACKEND_RNG_DESCRIPTOR;
    case DescriptorType_t::OPERATION_RNG_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR;
    case DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR;
    case DescriptorType_t::OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR;
    case DescriptorType_t::OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR;
    // Upstream spells the same backend type two ways.
    case DescriptorType_t::OPERATION_CONCATENATE_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR;
    case DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR;
    case DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR;
    case DescriptorType_t::OPERATION_TRANSPOSE_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR;
    case DescriptorType_t::OPERATION_SLICE_DESCRIPTOR:
        return CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR;
    case DescriptorType_t::NOT_SET:
    default:
        return std::nullopt;
    }
}

// toFe covers every declared cuDNN value (hipDNN's FE enums are name
// supersets), so whether hipDNN can take the value is the FE -> hipDNN hop.
inline bool hasHipdnnCounterpart(DataType_t value)
{
    return ::hipdnn_frontend::toHipdnnDataType(value).has_value();
}

inline bool hasHipdnnCounterpart(ConvolutionMode_t value)
{
    return ::hipdnn_frontend::toBackendConvMode(value).has_value();
}

inline bool hasHipdnnCounterpart(PointwiseMode_t value)
{
    return ::hipdnn_frontend::toBackendPointwiseMode(value).has_value();
}

inline bool hasHipdnnCounterpart(ReductionMode_t value)
{
    return ::hipdnn_frontend::toBackendReductionMode(value).has_value();
}

inline bool hasHipdnnCounterpart(ResampleMode_t value)
{
    return ::hipdnn_frontend::toBackendResampleMode(value).has_value();
}

inline bool hasHipdnnCounterpart(PaddingMode_t value)
{
    return ::hipdnn_frontend::toBackendPaddingMode(value).has_value();
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend

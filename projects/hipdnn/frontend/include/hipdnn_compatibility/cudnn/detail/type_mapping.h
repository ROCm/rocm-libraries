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

#include <cstddef>
#include <optional>

#include <hipdnn_compatibility/cudnn/detail/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>

// cuDNN C enum <-> shim FE enum, matched by enumerator name as upstream's
// convert_to_cudnn_type / convert_from_cudnn_type do. std::nullopt means the
// value is valid on one side but has no counterpart on the other; callers
// report NOT_SUPPORTED. Raw C values outside the declared range never reach
// these functions: the builders and the v8 factories reject them with BAD_PARAM.
// FE -> hipDNN C lowering reuses the Types.hpp converters (toHipdnnDataType,
// toBackendPointwiseMode, toBackendConvMode, ...).
namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

template <typename C, typename Fe>
struct EnumPair
{
    C cudnn;
    Fe fe;
};

// One table per enum pair. Lookups return the first matching row, so an alias
// row placed after its canonical row maps only to the canonical value.
// The initializer sizes each table, which std::array cannot deduce in C++17.
// NOLINTBEGIN(modernize-avoid-c-arrays)
template <typename C, typename Fe, std::size_t N>
constexpr std::optional<Fe> findFe(const EnumPair<C, Fe> (&table)[N], C value) noexcept
{
    for(const EnumPair<C, Fe>& row : table)
    {
        if(row.cudnn == value)
        {
            return row.fe;
        }
    }
    return std::nullopt;
}

template <typename C, typename Fe, std::size_t N>
constexpr std::optional<C> findCudnn(const EnumPair<C, Fe> (&table)[N], Fe value) noexcept
{
    for(const EnumPair<C, Fe>& row : table)
    {
        if(row.fe == value)
        {
            return row.cudnn;
        }
    }
    return std::nullopt;
}

inline constexpr EnumPair<cudnnDataType_t, DataType_t> DATA_TYPE_MAP[] = {
    {CUDNN_DATA_BFLOAT16, DataType_t::BFLOAT16},
    {CUDNN_DATA_BOOLEAN, DataType_t::BOOLEAN},
    {CUDNN_DATA_COMPLEX_FP32, DataType_t::COMPLEX_FP32},
    {CUDNN_DATA_COMPLEX_FP64, DataType_t::COMPLEX_FP64},
    {CUDNN_DATA_DOUBLE, DataType_t::DOUBLE},
    {CUDNN_DATA_FAST_FLOAT_FOR_FP8, DataType_t::FAST_FLOAT_FOR_FP8},
    {CUDNN_DATA_FLOAT, DataType_t::FLOAT},
    {CUDNN_DATA_FP4_E2M1, DataType_t::FP4_E2M1},
    {CUDNN_DATA_FP8_E4M3, DataType_t::FP8_E4M3},
    {CUDNN_DATA_FP8_E5M2, DataType_t::FP8_E5M2},
    {CUDNN_DATA_FP8_E8M0, DataType_t::FP8_E8M0},
    {CUDNN_DATA_HALF, DataType_t::HALF},
    {CUDNN_DATA_INT32, DataType_t::INT32},
    {CUDNN_DATA_INT4, DataType_t::INT4},
    {CUDNN_DATA_INT64, DataType_t::INT64},
    {CUDNN_DATA_INT8, DataType_t::INT8},
    {CUDNN_DATA_INT8x32, DataType_t::INT8x32},
    {CUDNN_DATA_INT8x4, DataType_t::INT8x4},
    {CUDNN_DATA_UINT8, DataType_t::UINT8},
    {CUDNN_DATA_UINT8x4, DataType_t::UINT8x4},
};

inline constexpr EnumPair<cudnnPointwiseMode_t, PointwiseMode_t> POINTWISE_MODE_MAP[] = {
    {CUDNN_POINTWISE_ABS, PointwiseMode_t::ABS},
    {CUDNN_POINTWISE_ADD, PointwiseMode_t::ADD},
    {CUDNN_POINTWISE_ADD_SQUARE, PointwiseMode_t::ADD_SQUARE},
    {CUDNN_POINTWISE_BINARY_SELECT, PointwiseMode_t::BINARY_SELECT},
    {CUDNN_POINTWISE_CEIL, PointwiseMode_t::CEIL},
    {CUDNN_POINTWISE_CMP_EQ, PointwiseMode_t::CMP_EQ},
    {CUDNN_POINTWISE_CMP_GE, PointwiseMode_t::CMP_GE},
    {CUDNN_POINTWISE_CMP_GT, PointwiseMode_t::CMP_GT},
    {CUDNN_POINTWISE_CMP_LE, PointwiseMode_t::CMP_LE},
    {CUDNN_POINTWISE_CMP_LT, PointwiseMode_t::CMP_LT},
    {CUDNN_POINTWISE_CMP_NEQ, PointwiseMode_t::CMP_NEQ},
    {CUDNN_POINTWISE_COS, PointwiseMode_t::COS},
    {CUDNN_POINTWISE_DIV, PointwiseMode_t::DIV},
    {CUDNN_POINTWISE_ELU_BWD, PointwiseMode_t::ELU_BWD},
    {CUDNN_POINTWISE_ELU_FWD, PointwiseMode_t::ELU_FWD},
    {CUDNN_POINTWISE_ERF, PointwiseMode_t::ERF},
    {CUDNN_POINTWISE_EXP, PointwiseMode_t::EXP},
    {CUDNN_POINTWISE_FLOOR, PointwiseMode_t::FLOOR},
    {CUDNN_POINTWISE_GELU_APPROX_TANH_BWD, PointwiseMode_t::GELU_APPROX_TANH_BWD},
    {CUDNN_POINTWISE_GELU_APPROX_TANH_FWD, PointwiseMode_t::GELU_APPROX_TANH_FWD},
    {CUDNN_POINTWISE_GELU_BWD, PointwiseMode_t::GELU_BWD},
    {CUDNN_POINTWISE_GELU_FWD, PointwiseMode_t::GELU_FWD},
    {CUDNN_POINTWISE_GEN_INDEX, PointwiseMode_t::GEN_INDEX},
    {CUDNN_POINTWISE_IDENTITY, PointwiseMode_t::IDENTITY},
    {CUDNN_POINTWISE_LOG, PointwiseMode_t::LOG},
    {CUDNN_POINTWISE_LOGICAL_AND, PointwiseMode_t::LOGICAL_AND},
    {CUDNN_POINTWISE_LOGICAL_NOT, PointwiseMode_t::LOGICAL_NOT},
    {CUDNN_POINTWISE_LOGICAL_OR, PointwiseMode_t::LOGICAL_OR},
    {CUDNN_POINTWISE_MAX, PointwiseMode_t::MAX},
    {CUDNN_POINTWISE_MIN, PointwiseMode_t::MIN},
    {CUDNN_POINTWISE_MOD, PointwiseMode_t::MOD},
    {CUDNN_POINTWISE_MUL, PointwiseMode_t::MUL},
    {CUDNN_POINTWISE_NEG, PointwiseMode_t::NEG},
    {CUDNN_POINTWISE_POW, PointwiseMode_t::POW},
    {CUDNN_POINTWISE_RECIPROCAL, PointwiseMode_t::RECIPROCAL},
    {CUDNN_POINTWISE_RELU_BWD, PointwiseMode_t::RELU_BWD},
    {CUDNN_POINTWISE_RELU_FWD, PointwiseMode_t::RELU_FWD},
    {CUDNN_POINTWISE_RSQRT, PointwiseMode_t::RSQRT},
    {CUDNN_POINTWISE_SIGMOID_BWD, PointwiseMode_t::SIGMOID_BWD},
    {CUDNN_POINTWISE_SIGMOID_FWD, PointwiseMode_t::SIGMOID_FWD},
    {CUDNN_POINTWISE_SIN, PointwiseMode_t::SIN},
    {CUDNN_POINTWISE_SOFTPLUS_BWD, PointwiseMode_t::SOFTPLUS_BWD},
    {CUDNN_POINTWISE_SOFTPLUS_FWD, PointwiseMode_t::SOFTPLUS_FWD},
    {CUDNN_POINTWISE_SQRT, PointwiseMode_t::SQRT},
    {CUDNN_POINTWISE_SUB, PointwiseMode_t::SUB},
    {CUDNN_POINTWISE_SWISH_BWD, PointwiseMode_t::SWISH_BWD},
    {CUDNN_POINTWISE_SWISH_FWD, PointwiseMode_t::SWISH_FWD},
    {CUDNN_POINTWISE_TAN, PointwiseMode_t::TAN},
    {CUDNN_POINTWISE_TANH_BWD, PointwiseMode_t::TANH_BWD},
    {CUDNN_POINTWISE_TANH_FWD, PointwiseMode_t::TANH_FWD},
};

inline constexpr EnumPair<cudnnConvolutionMode_t, ConvolutionMode_t> CONVOLUTION_MODE_MAP[] = {
    {CUDNN_CONVOLUTION, ConvolutionMode_t::CONVOLUTION},
    {CUDNN_CROSS_CORRELATION, ConvolutionMode_t::CROSS_CORRELATION},
};

inline constexpr EnumPair<cudnnReduceTensorOp_t, ReductionMode_t> REDUCTION_MODE_MAP[] = {
    {CUDNN_REDUCE_TENSOR_ADD, ReductionMode_t::ADD},
    {CUDNN_REDUCE_TENSOR_AMAX, ReductionMode_t::AMAX},
    {CUDNN_REDUCE_TENSOR_AVG, ReductionMode_t::AVG},
    {CUDNN_REDUCE_TENSOR_MAX, ReductionMode_t::MAX},
    {CUDNN_REDUCE_TENSOR_MIN, ReductionMode_t::MIN},
    {CUDNN_REDUCE_TENSOR_MUL, ReductionMode_t::MUL},
    {CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS, ReductionMode_t::MUL_NO_ZEROS},
    {CUDNN_REDUCE_TENSOR_NORM1, ReductionMode_t::NORM1},
    {CUDNN_REDUCE_TENSOR_NORM2, ReductionMode_t::NORM2},
};

inline constexpr EnumPair<cudnnResampleMode_t, ResampleMode_t> RESAMPLE_MODE_MAP[] = {
    {CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING, ResampleMode_t::AVGPOOL_EXCLUDE_PADDING},
    {CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING, ResampleMode_t::AVGPOOL_INCLUDE_PADDING},
    {CUDNN_RESAMPLE_BILINEAR, ResampleMode_t::BILINEAR},
    {CUDNN_RESAMPLE_MAXPOOL, ResampleMode_t::MAXPOOL},
    {CUDNN_RESAMPLE_NEAREST, ResampleMode_t::NEAREST},
};

inline constexpr EnumPair<cudnnPaddingMode_t, PaddingMode_t> PADDING_MODE_MAP[] = {
    {CUDNN_EDGE_VAL_PAD, PaddingMode_t::EDGE_VAL_PAD},
    {CUDNN_NEG_INF_PAD, PaddingMode_t::NEG_INF_PAD},
    {CUDNN_ZERO_PAD, PaddingMode_t::ZERO_PAD},
};

inline constexpr EnumPair<cudnnBackendNormFwdPhase_t, NormFwdPhase_t> NORM_FWD_PHASE_MAP[] = {
    {CUDNN_NORM_FWD_INFERENCE, NormFwdPhase_t::INFERENCE},
    {CUDNN_NORM_FWD_TRAINING, NormFwdPhase_t::TRAINING},
};

inline constexpr EnumPair<cudnnBackendNormMode_t, NormMode_t> NORM_MODE_MAP[] = {
    {CUDNN_ADA_LAYER_NORM, NormMode_t::ADA_LAYER_NORM},
    {CUDNN_BATCH_NORM, NormMode_t::BATCH_NORM},
    {CUDNN_GROUP_NORM, NormMode_t::GROUP_NORM},
    {CUDNN_INSTANCE_NORM, NormMode_t::INSTANCE_NORM},
    {CUDNN_LAYER_NORM, NormMode_t::LAYER_NORM},
    {CUDNN_RMS_NORM, NormMode_t::RMS_NORM},
};

inline constexpr EnumPair<cudnnBackendBehaviorNote_t, BehaviorNote_t> BEHAVIOR_NOTE_MAP[] = {
    {CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY, BehaviorNote_t::CUBLASLT_DEPENDENCY},
    {CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER,
     BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER},
    {CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER,
     BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER},
    {CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION, BehaviorNote_t::RUNTIME_COMPILATION},
    {CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API,
     BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API},
};

inline constexpr EnumPair<cudnnBackendNumericalNote_t, NumericalNote_t> NUMERICAL_NOTE_MAP[] = {
    {CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS, NumericalNote_t::DOWN_CONVERT_INPUTS},
    {CUDNN_NUMERICAL_NOTE_FFT, NumericalNote_t::FFT},
    {CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC, NumericalNote_t::NONDETERMINISTIC},
    {CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION,
     NumericalNote_t::REDUCED_PRECISION_REDUCTION},
    {CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP, NumericalNote_t::STRICT_NAN_PROP},
    {CUDNN_NUMERICAL_NOTE_TENSOR_CORE, NumericalNote_t::TENSOR_CORE},
    {CUDNN_NUMERICAL_NOTE_WINOGRAD, NumericalNote_t::WINOGRAD},
    {CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13, NumericalNote_t::WINOGRAD_TILE_13x13},
    {CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4, NumericalNote_t::WINOGRAD_TILE_4x4},
    {CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6, NumericalNote_t::WINOGRAD_TILE_6x6},
};

inline constexpr EnumPair<cudnnBackendTensorReordering_t, TensorReordering_t>
    TENSOR_REORDERING_MAP[] = {
        {CUDNN_TENSOR_REORDERING_NONE, TensorReordering_t::NONE},
        {CUDNN_TENSOR_REORDERING_INT8x32, TensorReordering_t::INT8x32},
        {CUDNN_TENSOR_REORDERING_F16x16, TensorReordering_t::F16x16},
        {CUDNN_TENSOR_REORDERING_F8_128x4, TensorReordering_t::F8_128x4},
};

inline constexpr EnumPair<cudnnBackendReshapeMode_t, ReshapeMode_t> RESHAPE_MODE_MAP[] = {
    {CUDNN_RESHAPE_LOGICAL, ReshapeMode_t::LOGICAL},
    {CUDNN_RESHAPE_VIEW_ONLY, ReshapeMode_t::VIEW_ONLY},
};

inline constexpr EnumPair<cudnnBackendKnobType_t, KnobType_t> KNOB_TYPE_MAP[] = {
    {CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD, KnobType_t::ARRAY_SIZE_PER_THREAD},
    {CUDNN_KNOB_TYPE_BLOCK_SIZE, KnobType_t::BLOCK_SIZE},
    {CUDNN_KNOB_TYPE_CTA_COUNT, KnobType_t::CTA_COUNT},
    {CUDNN_KNOB_TYPE_EDGE, KnobType_t::EDGE},
    {CUDNN_KNOB_TYPE_IDX_MODE, KnobType_t::IDX_MODE},
    {CUDNN_KNOB_TYPE_KERNEL_CFG, KnobType_t::KERNEL_CFG},
    {CUDNN_KNOB_TYPE_LOAD_SIZE, KnobType_t::LOAD_SIZE},
    {CUDNN_KNOB_TYPE_MULTIPLY, KnobType_t::MULTIPLY},
    {CUDNN_KNOB_TYPE_OCCUPANCY, KnobType_t::OCCUPANCY},
    {CUDNN_KNOB_TYPE_REDUCTION_MODE, KnobType_t::REDUCTION_MODE},
    {CUDNN_KNOB_TYPE_SPECFILT, KnobType_t::SPECFILT},
    {CUDNN_KNOB_TYPE_SPLIT_COLS, KnobType_t::SPLIT_COLS},
    {CUDNN_KNOB_TYPE_SPLIT_K_BUF, KnobType_t::SPLIT_K_BUF},
    {CUDNN_KNOB_TYPE_SPLIT_K_SLC, KnobType_t::SPLIT_K_SLC},
    {CUDNN_KNOB_TYPE_SPLIT_P_SLC, KnobType_t::SPLIT_P_SLC},
    {CUDNN_KNOB_TYPE_STAGES, KnobType_t::STAGES},
    {CUDNN_KNOB_TYPE_STREAM_K, KnobType_t::STREAM_K},
    {CUDNN_KNOB_TYPE_SWIZZLE, KnobType_t::SWIZZLE},
    {CUDNN_KNOB_TYPE_TILEK, KnobType_t::TILEK},
    {CUDNN_KNOB_TYPE_TILE_CGA_M, KnobType_t::TILE_CGA_M},
    {CUDNN_KNOB_TYPE_TILE_CGA_N, KnobType_t::TILE_CGA_N},
    {CUDNN_KNOB_TYPE_TILE_COLS, KnobType_t::TILE_COLS},
    {CUDNN_KNOB_TYPE_TILE_M, KnobType_t::TILE_M},
    {CUDNN_KNOB_TYPE_TILE_N, KnobType_t::TILE_N},
    {CUDNN_KNOB_TYPE_TILE_ROWS, KnobType_t::TILE_ROWS},
    {CUDNN_KNOB_TYPE_TILE_SIZE, KnobType_t::TILE_SIZE},
    {CUDNN_KNOB_TYPE_WARP_SPEC_CFG, KnobType_t::WARP_SPEC_CFG},
    {CUDNN_KNOB_TYPE_WORKSPACE, KnobType_t::WORKSPACE},
};

inline constexpr EnumPair<cudnnRngDistribution_t, RngDistribution_t> RNG_DISTRIBUTION_MAP[] = {
    {CUDNN_RNG_DISTRIBUTION_BERNOULLI, RngDistribution_t::BERNOULLI},
    {CUDNN_RNG_DISTRIBUTION_NORMAL, RngDistribution_t::NORMAL},
    {CUDNN_RNG_DISTRIBUTION_UNIFORM, RngDistribution_t::UNIFORM},
};

inline constexpr EnumPair<cudnnBackendDescriptorType_t, DescriptorType_t> DESCRIPTOR_TYPE_MAP[] = {
    {CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, DescriptorType_t::CONVOLUTION_DESCRIPTOR},
    {CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, DescriptorType_t::ENGINECFG_DESCRIPTOR},
    {CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, DescriptorType_t::ENGINEHEUR_DESCRIPTOR},
    {CUDNN_BACKEND_ENGINE_DESCRIPTOR, DescriptorType_t::ENGINE_DESCRIPTOR},
    {CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR},
    {CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR, DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR},
    {CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, DescriptorType_t::KNOB_CHOICE_DESCRIPTOR},
    {CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR, DescriptorType_t::KNOB_INFO_DESCRIPTOR},
    {CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR, DescriptorType_t::LAYOUT_INFO_DESCRIPTOR},
    {CUDNN_BACKEND_MATMUL_DESCRIPTOR, DescriptorType_t::MATMUL_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR,
     DescriptorType_t::OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR,
     DescriptorType_t::OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
     DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
     DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR, DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR, DescriptorType_t::OPERATION_CONCATENATE_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
     DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
     DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
     DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
     DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR,
     DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR,
     DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
     DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
     DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
     DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
     DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
     DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
     DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
     DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR, DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR, DescriptorType_t::OPERATION_RNG_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR, DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR, DescriptorType_t::OPERATION_SLICE_DESCRIPTOR},
    {CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR,
     DescriptorType_t::OPERATION_TRANSPOSE_DESCRIPTOR},
    {CUDNN_BACKEND_POINTWISE_DESCRIPTOR, DescriptorType_t::POINTWISE_DESCRIPTOR},
    {CUDNN_BACKEND_REDUCTION_DESCRIPTOR, DescriptorType_t::REDUCTION_DESCRIPTOR},
    {CUDNN_BACKEND_RESAMPLE_DESCRIPTOR, DescriptorType_t::RESAMPLE_DESCRIPTOR},
    {CUDNN_BACKEND_RNG_DESCRIPTOR, DescriptorType_t::RNG_DESCRIPTOR},
    {CUDNN_BACKEND_TENSOR_DESCRIPTOR, DescriptorType_t::TENSOR_DESCRIPTOR},
    {CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, DescriptorType_t::VARIANT_PACK_DESCRIPTOR},
};
// NOLINTEND(modernize-avoid-c-arrays)

constexpr std::optional<DataType_t> toFe(cudnnDataType_t value) noexcept
{
    return findFe(DATA_TYPE_MAP, value);
}

constexpr std::optional<cudnnDataType_t> toCudnn(DataType_t value) noexcept
{
    return findCudnn(DATA_TYPE_MAP, value);
}

constexpr std::optional<PointwiseMode_t> toFe(cudnnPointwiseMode_t value) noexcept
{
    return findFe(POINTWISE_MODE_MAP, value);
}

constexpr std::optional<cudnnPointwiseMode_t> toCudnn(PointwiseMode_t value) noexcept
{
    return findCudnn(POINTWISE_MODE_MAP, value);
}

constexpr std::optional<ConvolutionMode_t> toFe(cudnnConvolutionMode_t value) noexcept
{
    return findFe(CONVOLUTION_MODE_MAP, value);
}

constexpr std::optional<cudnnConvolutionMode_t> toCudnn(ConvolutionMode_t value) noexcept
{
    return findCudnn(CONVOLUTION_MODE_MAP, value);
}

constexpr std::optional<ReductionMode_t> toFe(cudnnReduceTensorOp_t value) noexcept
{
    return findFe(REDUCTION_MODE_MAP, value);
}

constexpr std::optional<cudnnReduceTensorOp_t> toCudnn(ReductionMode_t value) noexcept
{
    return findCudnn(REDUCTION_MODE_MAP, value);
}

constexpr std::optional<ResampleMode_t> toFe(cudnnResampleMode_t value) noexcept
{
    return findFe(RESAMPLE_MODE_MAP, value);
}

constexpr std::optional<cudnnResampleMode_t> toCudnn(ResampleMode_t value) noexcept
{
    return findCudnn(RESAMPLE_MODE_MAP, value);
}

constexpr std::optional<PaddingMode_t> toFe(cudnnPaddingMode_t value) noexcept
{
    return findFe(PADDING_MODE_MAP, value);
}

constexpr std::optional<cudnnPaddingMode_t> toCudnn(PaddingMode_t value) noexcept
{
    return findCudnn(PADDING_MODE_MAP, value);
}

constexpr std::optional<NormFwdPhase_t> toFe(cudnnBackendNormFwdPhase_t value) noexcept
{
    return findFe(NORM_FWD_PHASE_MAP, value);
}

constexpr std::optional<cudnnBackendNormFwdPhase_t> toCudnn(NormFwdPhase_t value) noexcept
{
    return findCudnn(NORM_FWD_PHASE_MAP, value);
}

constexpr std::optional<NormMode_t> toFe(cudnnBackendNormMode_t value) noexcept
{
    return findFe(NORM_MODE_MAP, value);
}

constexpr std::optional<cudnnBackendNormMode_t> toCudnn(NormMode_t value) noexcept
{
    return findCudnn(NORM_MODE_MAP, value);
}

constexpr std::optional<BehaviorNote_t> toFe(cudnnBackendBehaviorNote_t value) noexcept
{
    return findFe(BEHAVIOR_NOTE_MAP, value);
}

constexpr std::optional<cudnnBackendBehaviorNote_t> toCudnn(BehaviorNote_t value) noexcept
{
    return findCudnn(BEHAVIOR_NOTE_MAP, value);
}

constexpr std::optional<NumericalNote_t> toFe(cudnnBackendNumericalNote_t value) noexcept
{
    return findFe(NUMERICAL_NOTE_MAP, value);
}

constexpr std::optional<cudnnBackendNumericalNote_t> toCudnn(NumericalNote_t value) noexcept
{
    return findCudnn(NUMERICAL_NOTE_MAP, value);
}

constexpr std::optional<TensorReordering_t> toFe(cudnnBackendTensorReordering_t value) noexcept
{
    return findFe(TENSOR_REORDERING_MAP, value);
}

constexpr std::optional<cudnnBackendTensorReordering_t> toCudnn(TensorReordering_t value) noexcept
{
    return findCudnn(TENSOR_REORDERING_MAP, value);
}

constexpr std::optional<ReshapeMode_t> toFe(cudnnBackendReshapeMode_t value) noexcept
{
    return findFe(RESHAPE_MODE_MAP, value);
}

constexpr std::optional<cudnnBackendReshapeMode_t> toCudnn(ReshapeMode_t value) noexcept
{
    return findCudnn(RESHAPE_MODE_MAP, value);
}

constexpr std::optional<KnobType_t> toFe(cudnnBackendKnobType_t value) noexcept
{
    return findFe(KNOB_TYPE_MAP, value);
}

constexpr std::optional<cudnnBackendKnobType_t> toCudnn(KnobType_t value) noexcept
{
    return findCudnn(KNOB_TYPE_MAP, value);
}

constexpr std::optional<RngDistribution_t> toFe(cudnnRngDistribution_t value) noexcept
{
    return findFe(RNG_DISTRIBUTION_MAP, value);
}

constexpr std::optional<cudnnRngDistribution_t> toCudnn(RngDistribution_t value) noexcept
{
    return findCudnn(RNG_DISTRIBUTION_MAP, value);
}

constexpr std::optional<DescriptorType_t> toFe(cudnnBackendDescriptorType_t value) noexcept
{
    return findFe(DESCRIPTOR_TYPE_MAP, value);
}

constexpr std::optional<cudnnBackendDescriptorType_t> toCudnn(DescriptorType_t value) noexcept
{
    return findCudnn(DESCRIPTOR_TYPE_MAP, value);
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

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail

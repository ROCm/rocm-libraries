// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Declares identifiers matching the NVIDIA cuDNN C API, for source compatibility.
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_*.h, include/cudnn_backend_base.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//   include/cudnn_frontend/knobs.h
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file c_types.h
 * @brief cuDNN backend C-API types for the hipDNN cuDNN v8 compatibility shim.
 *
 * Enumerator values are shim-assigned, not NVIDIA's, and append-only; they are
 * pinned by tests/data/cudnn_v8/cudnn_v8_enum_values.inc.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <hipdnn_backend.h>
#include <hipdnn_compatibility/cudnn/cudnn_status.h>

/// @brief cuDNN handle, aliased directly to the hipDNN handle type.
///
/// `hipdnnHandle_t` is the global C typedef from `<hipdnn_backend.h>`, not a
/// member of namespace `hipdnn_frontend`.
using cudnnHandle_t = ::hipdnnHandle_t;

static_assert(std::is_same_v<cudnnHandle_t, ::hipdnnHandle_t>,
              "cudnnHandle_t must alias the hipDNN handle type");

// Upstream sizes fixed arrays with CUDNN_DIM_MAX + 1.
#define CUDNN_DIM_MAX 8 // NOLINT(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)

// NOLINTBEGIN(readability-identifier-naming)

/// @brief Backend descriptor, aliased to the hipDNN descriptor type so plans and variant packs
/// pass straight to hipDNN.
using cudnnBackendDescriptor_t = ::hipdnnBackendDescriptor_t;

/// @brief Rational number attribute value. Mirrors NVIDIA `cudnnFraction_t`.
typedef struct
{
    std::int64_t numerator;
    std::int64_t denominator;
} cudnnFraction_t;

/// @brief Mirrors NVIDIA `cudnnBackendBehaviorNote_t`.
typedef enum
{
    CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY = 0,
    CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER = 1,
    CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER = 2,
    CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION = 3,
    CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API = 4,
    CUDNN_BEHAVIOR_NOTE_TYPE_COUNT = 5,
} cudnnBackendBehaviorNote_t;
static_assert(sizeof(cudnnBackendBehaviorNote_t) == 4,
              "cudnnBackendBehaviorNote_t must be 4 bytes");
static_assert(CUDNN_BEHAVIOR_NOTE_TYPE_COUNT
                  == CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API + 1,
              "CUDNN_BEHAVIOR_NOTE_TYPE_COUNT must be last + 1");

/// @brief Mirrors NVIDIA `cudnnBackendDescriptorType_t`.
typedef enum
{
    CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR = 0,
    CUDNN_BACKEND_ENGINECFG_DESCRIPTOR = 1,
    CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR = 2,
    CUDNN_BACKEND_ENGINE_DESCRIPTOR = 3,
    CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR = 4,
    CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR = 5,
    CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR = 6,
    CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR = 7,
    CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR = 8,
    CUDNN_BACKEND_MATMUL_DESCRIPTOR = 9,
    CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR = 10,
    CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR = 11,
    CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR = 12,
    CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR = 13,
    CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR = 14,
    CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR = 15,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = 16,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = 17,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = 18,
    CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = 19,
    CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = 20,
    CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR = 21,
    CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR = 22,
    CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = 23,
    CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = 24,
    CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR = 25,
    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = 26,
    CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = 27,
    CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = 28,
    CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = 29,
    CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR = 30,
    CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR = 31,
    CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR = 32,
    CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR = 33,
    CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR = 34,
    CUDNN_BACKEND_POINTWISE_DESCRIPTOR = 35,
    CUDNN_BACKEND_REDUCTION_DESCRIPTOR = 36,
    CUDNN_BACKEND_RESAMPLE_DESCRIPTOR = 37,
    CUDNN_BACKEND_RNG_DESCRIPTOR = 38,
    CUDNN_BACKEND_TENSOR_DESCRIPTOR = 39,
    CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = 40,
} cudnnBackendDescriptorType_t;
static_assert(sizeof(cudnnBackendDescriptorType_t) == 4,
              "cudnnBackendDescriptorType_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnBackendHeurMode_t`.
typedef enum
{
    CUDNN_HEUR_MODE_A = 0,
    CUDNN_HEUR_MODE_B = 1,
    CUDNN_HEUR_MODE_FALLBACK = 2,
    CUDNN_HEUR_MODE_INSTANT = 3,
    CUDNN_HEUR_MODES_COUNT = 4,
} cudnnBackendHeurMode_t;
static_assert(sizeof(cudnnBackendHeurMode_t) == 4, "cudnnBackendHeurMode_t must be 4 bytes");
static_assert(CUDNN_HEUR_MODES_COUNT == CUDNN_HEUR_MODE_INSTANT + 1,
              "CUDNN_HEUR_MODES_COUNT must be last + 1");

/// @brief Mirrors NVIDIA `cudnnBackendKnobType_t`.
typedef enum
{
    CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD = 0,
    CUDNN_KNOB_TYPE_BLOCK_SIZE = 1,
    CUDNN_KNOB_TYPE_CTA_COUNT = 2,
    CUDNN_KNOB_TYPE_EDGE = 3,
    CUDNN_KNOB_TYPE_IDX_MODE = 4,
    CUDNN_KNOB_TYPE_KERNEL_CFG = 5,
    CUDNN_KNOB_TYPE_LOAD_SIZE = 6,
    CUDNN_KNOB_TYPE_MULTIPLY = 7,
    CUDNN_KNOB_TYPE_OCCUPANCY = 8,
    CUDNN_KNOB_TYPE_REDUCTION_MODE = 9,
    CUDNN_KNOB_TYPE_SPECFILT = 10,
    CUDNN_KNOB_TYPE_SPLIT_COLS = 11,
    CUDNN_KNOB_TYPE_SPLIT_K_BUF = 12,
    CUDNN_KNOB_TYPE_SPLIT_K_SLC = 13,
    CUDNN_KNOB_TYPE_SPLIT_P_SLC = 14,
    CUDNN_KNOB_TYPE_STAGES = 15,
    CUDNN_KNOB_TYPE_STREAM_K = 16,
    CUDNN_KNOB_TYPE_SWIZZLE = 17,
    CUDNN_KNOB_TYPE_TILEK = 18,
    CUDNN_KNOB_TYPE_TILE_CGA_M = 19,
    CUDNN_KNOB_TYPE_TILE_CGA_N = 20,
    CUDNN_KNOB_TYPE_TILE_COLS = 21,
    CUDNN_KNOB_TYPE_TILE_M = 22,
    CUDNN_KNOB_TYPE_TILE_N = 23,
    CUDNN_KNOB_TYPE_TILE_ROWS = 24,
    CUDNN_KNOB_TYPE_TILE_SIZE = 25,
    CUDNN_KNOB_TYPE_WARP_SPEC_CFG = 26,
    CUDNN_KNOB_TYPE_WORKSPACE = 27,
    CUDNN_KNOB_TYPE_COUNTS = 28,
} cudnnBackendKnobType_t;
static_assert(sizeof(cudnnBackendKnobType_t) == 4, "cudnnBackendKnobType_t must be 4 bytes");
static_assert(CUDNN_KNOB_TYPE_COUNTS == CUDNN_KNOB_TYPE_WORKSPACE + 1,
              "CUDNN_KNOB_TYPE_COUNTS must be last + 1");

/// @brief Mirrors NVIDIA `cudnnBackendNormFwdPhase_t`.
typedef enum
{
    CUDNN_NORM_FWD_INFERENCE = 0,
    CUDNN_NORM_FWD_TRAINING = 1,
} cudnnBackendNormFwdPhase_t;
static_assert(sizeof(cudnnBackendNormFwdPhase_t) == 4,
              "cudnnBackendNormFwdPhase_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnBackendNormMode_t`.
typedef enum
{
    CUDNN_ADA_LAYER_NORM = 0,
    CUDNN_BATCH_NORM = 1,
    CUDNN_GROUP_NORM = 2,
    CUDNN_INSTANCE_NORM = 3,
    CUDNN_LAYER_NORM = 4,
    CUDNN_RMS_NORM = 5,
} cudnnBackendNormMode_t;
static_assert(sizeof(cudnnBackendNormMode_t) == 4, "cudnnBackendNormMode_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnBackendNumericalNote_t`.
typedef enum
{
    CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS = 0,
    CUDNN_NUMERICAL_NOTE_FFT = 1,
    CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC = 2,
    CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION = 3,
    CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP = 4,
    CUDNN_NUMERICAL_NOTE_TENSOR_CORE = 5,
    CUDNN_NUMERICAL_NOTE_WINOGRAD = 6,
    CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13 = 7,
    CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4 = 8,
    CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6 = 9,
    CUDNN_NUMERICAL_NOTE_TYPE_COUNT = 10,
} cudnnBackendNumericalNote_t;
static_assert(sizeof(cudnnBackendNumericalNote_t) == 4,
              "cudnnBackendNumericalNote_t must be 4 bytes");
static_assert(CUDNN_NUMERICAL_NOTE_TYPE_COUNT == CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6 + 1,
              "CUDNN_NUMERICAL_NOTE_TYPE_COUNT must be last + 1");

/// @brief Mirrors NVIDIA `cudnnBackendReshapeMode_t`.
typedef enum
{
    CUDNN_RESHAPE_LOGICAL = 0,
    CUDNN_RESHAPE_VIEW_ONLY = 1,
} cudnnBackendReshapeMode_t;
static_assert(sizeof(cudnnBackendReshapeMode_t) == 4, "cudnnBackendReshapeMode_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnBackendTensorReordering_t`.
typedef enum
{
    CUDNN_TENSOR_REORDERING_NONE = 0,
    CUDNN_TENSOR_REORDERING_INT8x32 = 1,
    CUDNN_TENSOR_REORDERING_F16x16 = 2,
    CUDNN_TENSOR_REORDERING_F8_128x4 = 3,
} cudnnBackendTensorReordering_t;
static_assert(sizeof(cudnnBackendTensorReordering_t) == 4,
              "cudnnBackendTensorReordering_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnBnFinalizeStatsMode_t`.
typedef enum
{
    CUDNN_BN_FINALIZE_STATISTICS_TRAINING = 0,
} cudnnBnFinalizeStatsMode_t;
static_assert(sizeof(cudnnBnFinalizeStatsMode_t) == 4,
              "cudnnBnFinalizeStatsMode_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnConvolutionMode_t`.
typedef enum
{
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
} cudnnConvolutionMode_t;
static_assert(sizeof(cudnnConvolutionMode_t) == 4, "cudnnConvolutionMode_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnDataType_t`.
typedef enum
{
    CUDNN_DATA_BFLOAT16 = 0,
    CUDNN_DATA_BOOLEAN = 1,
    CUDNN_DATA_COMPLEX_FP32 = 2,
    CUDNN_DATA_COMPLEX_FP64 = 3,
    CUDNN_DATA_DOUBLE = 4,
    CUDNN_DATA_FAST_FLOAT_FOR_FP8 = 5,
    CUDNN_DATA_FLOAT = 6,
    CUDNN_DATA_FP4_E2M1 = 7,
    CUDNN_DATA_FP8_E4M3 = 8,
    CUDNN_DATA_FP8_E5M2 = 9,
    CUDNN_DATA_FP8_E8M0 = 10,
    CUDNN_DATA_HALF = 11,
    CUDNN_DATA_INT32 = 12,
    CUDNN_DATA_INT4 = 13,
    CUDNN_DATA_INT64 = 14,
    CUDNN_DATA_INT8 = 15,
    CUDNN_DATA_INT8x32 = 16,
    CUDNN_DATA_INT8x4 = 17,
    CUDNN_DATA_UINT8 = 18,
    CUDNN_DATA_UINT8x4 = 19,
} cudnnDataType_t;
static_assert(sizeof(cudnnDataType_t) == 4, "cudnnDataType_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnGenStatsMode_t`.
typedef enum
{
    CUDNN_GENSTATS_SUM_SQSUM = 0,
} cudnnGenStatsMode_t;
static_assert(sizeof(cudnnGenStatsMode_t) == 4, "cudnnGenStatsMode_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnNanPropagation_t`.
typedef enum
{
    CUDNN_PROPAGATE_NAN = 0,
} cudnnNanPropagation_t;
static_assert(sizeof(cudnnNanPropagation_t) == 4, "cudnnNanPropagation_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnPaddingMode_t`.
typedef enum
{
    CUDNN_EDGE_VAL_PAD = 0,
    CUDNN_NEG_INF_PAD = 1,
    CUDNN_ZERO_PAD = 2,
} cudnnPaddingMode_t;
static_assert(sizeof(cudnnPaddingMode_t) == 4, "cudnnPaddingMode_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnPointwiseMode_t`.
typedef enum
{
    CUDNN_POINTWISE_ABS = 0,
    CUDNN_POINTWISE_ADD = 1,
    CUDNN_POINTWISE_ADD_SQUARE = 2,
    CUDNN_POINTWISE_BINARY_SELECT = 3,
    CUDNN_POINTWISE_CEIL = 4,
    CUDNN_POINTWISE_CMP_EQ = 5,
    CUDNN_POINTWISE_CMP_GE = 6,
    CUDNN_POINTWISE_CMP_GT = 7,
    CUDNN_POINTWISE_CMP_LE = 8,
    CUDNN_POINTWISE_CMP_LT = 9,
    CUDNN_POINTWISE_CMP_NEQ = 10,
    CUDNN_POINTWISE_COS = 11,
    CUDNN_POINTWISE_DIV = 12,
    CUDNN_POINTWISE_ELU_BWD = 13,
    CUDNN_POINTWISE_ELU_FWD = 14,
    CUDNN_POINTWISE_ERF = 15,
    CUDNN_POINTWISE_EXP = 16,
    CUDNN_POINTWISE_FLOOR = 17,
    CUDNN_POINTWISE_GELU_APPROX_TANH_BWD = 18,
    CUDNN_POINTWISE_GELU_APPROX_TANH_FWD = 19,
    CUDNN_POINTWISE_GELU_BWD = 20,
    CUDNN_POINTWISE_GELU_FWD = 21,
    CUDNN_POINTWISE_GEN_INDEX = 22,
    CUDNN_POINTWISE_IDENTITY = 23,
    CUDNN_POINTWISE_LOG = 24,
    CUDNN_POINTWISE_LOGICAL_AND = 25,
    CUDNN_POINTWISE_LOGICAL_NOT = 26,
    CUDNN_POINTWISE_LOGICAL_OR = 27,
    CUDNN_POINTWISE_MAX = 28,
    CUDNN_POINTWISE_MIN = 29,
    CUDNN_POINTWISE_MOD = 30,
    CUDNN_POINTWISE_MUL = 31,
    CUDNN_POINTWISE_NEG = 32,
    CUDNN_POINTWISE_POW = 33,
    CUDNN_POINTWISE_RECIPROCAL = 34,
    CUDNN_POINTWISE_RELU_BWD = 35,
    CUDNN_POINTWISE_RELU_FWD = 36,
    CUDNN_POINTWISE_RSQRT = 37,
    CUDNN_POINTWISE_SIGMOID_BWD = 38,
    CUDNN_POINTWISE_SIGMOID_FWD = 39,
    CUDNN_POINTWISE_SIN = 40,
    CUDNN_POINTWISE_SOFTPLUS_BWD = 41,
    CUDNN_POINTWISE_SOFTPLUS_FWD = 42,
    CUDNN_POINTWISE_SQRT = 43,
    CUDNN_POINTWISE_SUB = 44,
    CUDNN_POINTWISE_SWISH_BWD = 45,
    CUDNN_POINTWISE_SWISH_FWD = 46,
    CUDNN_POINTWISE_TAN = 47,
    CUDNN_POINTWISE_TANH_BWD = 48,
    CUDNN_POINTWISE_TANH_FWD = 49,
} cudnnPointwiseMode_t;
static_assert(sizeof(cudnnPointwiseMode_t) == 4, "cudnnPointwiseMode_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnReduceTensorOp_t`.
typedef enum
{
    CUDNN_REDUCE_TENSOR_ADD = 0,
    CUDNN_REDUCE_TENSOR_AMAX = 1,
    CUDNN_REDUCE_TENSOR_AVG = 2,
    CUDNN_REDUCE_TENSOR_MAX = 3,
    CUDNN_REDUCE_TENSOR_MIN = 4,
    CUDNN_REDUCE_TENSOR_MUL = 5,
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 6,
    CUDNN_REDUCE_TENSOR_NORM1 = 7,
    CUDNN_REDUCE_TENSOR_NORM2 = 8,
} cudnnReduceTensorOp_t;
static_assert(sizeof(cudnnReduceTensorOp_t) == 4, "cudnnReduceTensorOp_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnResampleMode_t`.
typedef enum
{
    CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING = 0,
    CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING = 1,
    CUDNN_RESAMPLE_BILINEAR = 2,
    CUDNN_RESAMPLE_MAXPOOL = 3,
    CUDNN_RESAMPLE_NEAREST = 4,
    CUDNN_RESAMPLE_AVGPOOL = CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING,
} cudnnResampleMode_t;
static_assert(sizeof(cudnnResampleMode_t) == 4, "cudnnResampleMode_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnRngDistribution_t`.
typedef enum
{
    CUDNN_RNG_DISTRIBUTION_BERNOULLI = 0,
    CUDNN_RNG_DISTRIBUTION_NORMAL = 1,
    CUDNN_RNG_DISTRIBUTION_UNIFORM = 2,
} cudnnRngDistribution_t;
static_assert(sizeof(cudnnRngDistribution_t) == 4, "cudnnRngDistribution_t must be 4 bytes");

/// @brief Mirrors NVIDIA `cudnnTensorFormat_t`.
typedef enum
{
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NCHW_VECT_C = 1,
    CUDNN_TENSOR_NHWC = 2,
} cudnnTensorFormat_t;
static_assert(sizeof(cudnnTensorFormat_t) == 4, "cudnnTensorFormat_t must be 4 bytes");

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

// Largest value of each enum above; the shim rejects anything past it as
// BAD_PARAM.
template <typename CEnum>
inline constexpr int LAST_ENUMERATOR = -1;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBackendBehaviorNote_t>
    = CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBackendDescriptorType_t>
    = CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBackendHeurMode_t> = CUDNN_HEUR_MODE_INSTANT;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBackendKnobType_t> = CUDNN_KNOB_TYPE_WORKSPACE;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBackendNormFwdPhase_t> = CUDNN_NORM_FWD_TRAINING;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBackendNormMode_t> = CUDNN_RMS_NORM;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBackendNumericalNote_t>
    = CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBackendReshapeMode_t> = CUDNN_RESHAPE_VIEW_ONLY;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBackendTensorReordering_t>
    = CUDNN_TENSOR_REORDERING_F8_128x4;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnBnFinalizeStatsMode_t>
    = CUDNN_BN_FINALIZE_STATISTICS_TRAINING;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnConvolutionMode_t> = CUDNN_CROSS_CORRELATION;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnDataType_t> = CUDNN_DATA_UINT8x4;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnGenStatsMode_t> = CUDNN_GENSTATS_SUM_SQSUM;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnNanPropagation_t> = CUDNN_PROPAGATE_NAN;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnPaddingMode_t> = CUDNN_ZERO_PAD;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnPointwiseMode_t> = CUDNN_POINTWISE_TANH_FWD;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnReduceTensorOp_t> = CUDNN_REDUCE_TENSOR_NORM2;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnResampleMode_t> = CUDNN_RESAMPLE_NEAREST;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnRngDistribution_t> = CUDNN_RNG_DISTRIBUTION_UNIFORM;

template <>
inline constexpr int LAST_ENUMERATOR<cudnnTensorFormat_t> = CUDNN_TENSOR_NHWC;

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail

// NOLINTEND(readability-identifier-naming)

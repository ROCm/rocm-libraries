// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file Types.hpp
 * @brief Core type definitions and enumerations for the hipDNN Frontend API
 *
 * Defines the fundamental enums that configure hipDNN operations:
 *
 * | hipDNN enum | Concept | Purpose |
 * |-------------|---------|---------|
 * | DataType | Numeric precision | fp32, fp16, bf16, fp8, etc. |
 * | PointwiseMode | Activation / element-wise op | relu, gelu, add, mul, etc. |
 * | ConvolutionMode | Filter application method | Cross-correlation vs mathematical convolution |
 * | NormFwdPhase | Training vs inference | Whether to save stats for backward pass |
 * | DiagonalAlignment | SDPA causal mask alignment | Top-left vs bottom-right diagonal |
 * | AttentionImplementation | SDPA execution strategy | Auto, composite, or unified kernel |
 * | HeuristicMode | Engine discovery mode | How engines are ranked |
 * | BuildPlanPolicy | Plan selection policy | Heuristic choice vs build all |
 * | KnobValueType | Engine knob data type | int64, float64, or string |
 *
 * This file also contains conversion utilities between frontend types and
 * the backend C API types (hipdnn_backend.h).
 *
 * Portions derived from NVIDIA cuDNN frontend, used under the MIT license.
 */

#pragma once

#include <HipdnnAttentionImplementation.h>
#include <HipdnnBackendBehaviorNote.h>
#include <HipdnnBackendHeuristicType.h>
#include <HipdnnConvolutionMode.h>
#include <HipdnnDataType.h>
#include <HipdnnDiagonalAlignment.h>
#include <HipdnnNormFwdPhase.h>
#include <HipdnnPaddingMode.h>
#include <HipdnnPointwiseMode.h>
#include <HipdnnReduceTensorOp.h>
#include <HipdnnResampleMode.h>
#include <hipdnn_data_sdk/types.hpp>

#include <hipdnn_frontend/Error.hpp>

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>

namespace hipdnn_frontend
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_data_sdk::types::fp8_e4m3_fnuz;
using hipdnn_data_sdk::types::fp8_e5m2;
using hipdnn_data_sdk::types::fp8_e5m2_fnuz;
using hipdnn_data_sdk::types::half;

/**
 * @enum ConvolutionMode
 * @brief Specifies the convolution algorithm mode
 *
 * Determines how the convolution filter is applied to the input tensor.
 */
enum class ConvolutionMode
{
    NOT_SET = 0, ///< Mode not specified
    CROSS_CORRELATION = 1, ///< Cross-correlation mode (standard deep learning convolution)
    CONVOLUTION = 2 ///< Mathematical convolution (filter is flipped)
};
typedef ConvolutionMode ConvolutionMode_t; ///< @brief Type alias for ConvolutionMode

/**
 * @enum PointwiseMode
 * @brief Specifies the type of element-wise (pointwise) operation
 *
 * Pointwise operations are applied element-by-element to tensors.
 * They can be unary (1 input), binary (2 inputs), or ternary (3 inputs).
 *
 * @see isUnaryPointwiseMode(), isBinaryPointwiseMode(), isTernaryPointwiseMode()
 */
enum class PointwiseMode
{
    NOT_SET = 0, ///< Mode not specified
    ABS = 1, ///< Absolute value: |x|
    ADD = 2, ///< Addition: x + y
    ADD_SQUARE = 3, ///< Add x to y squared: x + y²
    BINARY_SELECT = 4, ///< Ternary select based on condition
    CEIL = 5, ///< Ceiling function
    CMP_EQ = 6, ///< Compare equal: x == y
    CMP_GE = 7, ///< Compare greater or equal: x >= y
    CMP_GT = 8, ///< Compare greater than: x > y
    CMP_LE = 9, ///< Compare less or equal: x <= y
    CMP_LT = 10, ///< Compare less than: x < y
    CMP_NEQ = 11, ///< Compare not equal: x != y
    DIV = 12, ///< Division: x / y
    ELU_BWD = 13, ///< ELU activation backward pass
    ELU_FWD = 14, ///< ELU activation forward pass
    ERF = 15, ///< Error function
    EXP = 16, ///< Exponential: e^x
    FLOOR = 17, ///< Floor function
    GELU_APPROX_TANH_BWD = 18, ///< GELU (tanh approximation) backward
    GELU_APPROX_TANH_FWD = 19, ///< GELU (tanh approximation) forward
    GELU_BWD = 20, ///< GELU activation backward
    GELU_FWD = 21, ///< GELU activation forward
    GEN_INDEX = 22, ///< Generate index tensor
    IDENTITY = 23, ///< Identity: y = x
    LOG = 24, ///< Natural logarithm
    LOGICAL_AND = 25, ///< Logical AND
    LOGICAL_NOT = 26, ///< Logical NOT
    LOGICAL_OR = 27, ///< Logical OR
    MAX = 28, ///< Element-wise maximum
    MIN = 29, ///< Element-wise minimum
    MUL = 30, ///< Multiplication: x * y
    NEG = 31, ///< Negation: -x
    RECIPROCAL = 32, ///< Reciprocal: 1/x
    RELU_BWD = 33, ///< ReLU backward pass
    RELU_FWD = 34, ///< ReLU forward pass
    RSQRT = 35, ///< Reciprocal square root: 1/sqrt(x)
    SIGMOID_BWD = 36, ///< Sigmoid backward pass
    SIGMOID_FWD = 37, ///< Sigmoid forward pass
    SIN = 38, ///< Sine function
    SOFTPLUS_BWD = 39, ///< Softplus backward pass
    SOFTPLUS_FWD = 40, ///< Softplus forward pass
    SQRT = 41, ///< Square root
    SUB = 42, ///< Subtraction: x - y
    SWISH_BWD = 43, ///< Swish activation backward
    SWISH_FWD = 44, ///< Swish activation forward
    TAN = 45, ///< Tangent function
    TANH_BWD = 46, ///< Tanh backward pass
    TANH_FWD = 47, ///< Tanh forward pass
    MOD = 48, ///< Modulo: x mod y (binary)
    POW = 49, ///< Power: x raised to y (binary)
    COS = 50, ///< Cosine function (unary)
    COUNT = 51 ///< Number of pointwise modes (sentinel — not a valid mode)
};
typedef PointwiseMode PointwiseMode_t; ///< @brief Type alias for PointwiseMode

/**
 * @enum ReductionMode
 * @brief Specifies the reduction operation to perform
 *
 * Matches the cudnn-frontend ReductionMode_t enumeration for API compatibility.
 */
enum class ReductionMode
{
    NOT_SET = 0, ///< Reduction mode not specified
    ADD = 1, ///< Sum reduction
    MUL = 2, ///< Product reduction
    MIN = 3, ///< Minimum reduction
    MAX = 4, ///< Maximum reduction
    AMAX = 5, ///< Absolute maximum reduction
    AVG = 6, ///< Average reduction
    NORM1 = 7, ///< L1 norm reduction
    NORM2 = 8, ///< L2 norm reduction
    MUL_NO_ZEROS = 9, ///< Product reduction excluding zeros
};
typedef ReductionMode ReductionMode_t; ///< @brief Type alias for ReductionMode

/**
 * @enum ResampleMode
 * @brief Specifies the resample operation mode
 */
enum class ResampleMode
{
    NOT_SET = 0, ///< Resample mode not specified
    MAXPOOL = 1, ///< Maximum pooling
    AVGPOOL_EXCLUDE_PADDING = 2, ///< Average pooling (excludes padding from divisor)
    AVGPOOL_INCLUDE_PADDING = 3, ///< Average pooling (includes padding in divisor)
    BILINEAR = 4, ///< Bilinear resampling
    NEAREST = 5 ///< Nearest-neighbor resampling
};
typedef ResampleMode ResampleMode_t; ///< @brief Type alias for ResampleMode

/**
 * @enum PaddingMode
 * @brief Specifies the padding mode for resample operations
 */
enum class PaddingMode
{
    NOT_SET = 0, ///< Padding mode not specified
    NEG_INF_PAD = 1, ///< Pad with negative infinity
    ZERO_PAD = 2, ///< Pad with zeros
    EDGE_VAL_PAD = 3 ///< Pad with the edge value
};
typedef PaddingMode PaddingMode_t; ///< @brief Type alias for PaddingMode

/**
 * @enum DataType
 * @brief Specifies the data type for tensor elements
 *
 * Defines the numeric precision and format for tensor data. Different operations
 * may support different subsets of data types.
 */
enum class DataType
{
    NOT_SET = 0, ///< Data type not specified
    FLOAT = 1, ///< 32-bit floating point (fp32)
    HALF = 2, ///< 16-bit floating point (fp16, IEEE 754)
    BFLOAT16 = 3, ///< 16-bit brain floating point (bf16)
    DOUBLE = 4, ///< 64-bit floating point (fp64)
    UINT8 = 5, ///< 8-bit unsigned integer
    INT32 = 6, ///< 32-bit signed integer
    INT8 = 7, ///< 8-bit signed integer
    FP8_E4M3 = 8, ///< 8-bit floating point (4 exponent, 3 mantissa bits)
    FP8_E5M2 = 9, ///< 8-bit floating point (5 exponent, 2 mantissa bits)
    FP8_E8M0 = 10, ///< 8-bit floating point (8 exponent, 0 mantissa bits)
    FP4_E2M1 = 11, ///< 4-bit floating point (2 exponent, 1 mantissa bit)
    INT4 = 12, ///< 4-bit signed integer
    FP6_E2M3 = 13, ///< 6-bit floating point (2 exponent, 3 mantissa bits)
    FP6_E3M2 = 14, ///< 6-bit floating point (3 exponent, 2 mantissa bits)
    INT64 = 15, ///< 64-bit signed integer
    BOOLEAN = 16, ///< 8-bit boolean
    FP8_E4M3_FNUZ = 17, ///< 8-bit floating point (4 exponent, 3 mantissa bits, FNUZ)
    FP8_E5M2_FNUZ = 18, ///< 8-bit floating point (5 exponent, 2 mantissa bits, FNUZ)
    // NOLINTNEXTLINE(readability-identifier-naming)
    INT8x4 = 19, ///< Four packed 8-bit signed integers (vectorized layout)
    // NOLINTNEXTLINE(readability-identifier-naming)
    UINT8x4 = 20, ///< Four packed 8-bit unsigned integers (vectorized layout)
    // NOLINTNEXTLINE(readability-identifier-naming)
    INT8x32 = 21, ///< Thirty-two packed 8-bit signed integers (vectorized layout)
    FAST_FLOAT_FOR_FP8 = 22, ///< Fast floating-point accumulation type for FP8
    COMPLEX_FP32 = 23, ///< Complex number with 32-bit floating-point components
    COMPLEX_FP64 = 24, ///< Complex number with 64-bit floating-point components
};
typedef DataType DataType_t; ///< @brief Type alias for DataType

/**
 * @enum DiagonalAlignment
 * @brief Diagonal alignment mode for SDPA causal masking
 *
 * Controls which corner of the attention matrix the causal mask diagonal
 * is anchored to. This affects how the triangular mask is positioned
 * when query and key sequence lengths differ (S_q != S_kv).
 *
 * For equal-length sequences both modes produce the same mask. When
 * S_q < S_kv, BOTTOM_RIGHT aligns the mask so that each query attends
 * to the most recent keys rather than the earliest.
 *
 * @see hipdnn_frontend::graph::SdpaAttributes::set_diagonal_alignment()
 */
enum class DiagonalAlignment
{
    TOP_LEFT = 0, ///< Anchor mask diagonal to the top-left corner of the attention matrix
    BOTTOM_RIGHT = 1, ///< Anchor mask diagonal to the bottom-right corner of the attention matrix
};
// NOLINTNEXTLINE(readability-identifier-naming)
typedef DiagonalAlignment DiagonalAlignment_t; ///< @brief Type alias for DiagonalAlignment

/**
 * @enum AttentionImplementation
 * @brief Execution strategy for scaled dot-product attention
 *
 * Selects how the SDPA computation is mapped to GPU kernels.
 *
 * @see hipdnn_frontend::graph::SdpaAttributes::set_implementation()
 */
enum class AttentionImplementation
{
    AUTO = 0, ///< Let the backend choose the best strategy for the given configuration
    COMPOSITE = 1, ///< Decompose attention into separate matmul, softmax, and matmul kernels
    UNIFIED = 2, ///< Use a single fused kernel for the entire attention computation
};
// NOLINTNEXTLINE(readability-identifier-naming)
typedef AttentionImplementation
    AttentionImplementation_t; ///< @brief Type alias for AttentionImplementation

/**
 * @enum HeuristicMode
 * @brief Specifies the heuristic mode for engine selection
 *
 * Controls how the hipDNN backend selects execution plans and engines.
 */
enum class HeuristicMode
{
    FALLBACK, ///< Use fallback heuristics for engine selection
    A, ///< cuDNN heuristic mode A (mapped to fallback for now)
    B, ///< cuDNN heuristic mode B (mapped to fallback for now)
    OPENSOURCE, ///< cuDNN open-source heuristic mode (mapped to fallback for now)
};
typedef HeuristicMode HeurMode_t; ///< @brief Type alias for HeuristicMode

/**
 * @enum BehaviorNote
 * @brief Advisory behavior metadata reported by an engine
 */
enum class BehaviorNote : int32_t
{
    RUNTIME_COMPILATION = 0, ///< Engine may compile kernels or other code at runtime.
    REQUIRES_LAYOUT_TRANSFORM = 1, ///< Engine may require internal tensor layout transforms.
    SUPPORTS_GRAPH_CAPTURE = 2, ///< Engine supports execution during stream graph capture.
    EXTERNAL_LIBRARY_DEPENDENCY = 3, ///< Engine depends on a library outside core hipDNN.
    SUPPORTS_EXECUTION_PLAN_SERIALIZATION = 4 ///< Engine supports execution plan serialization.
};
typedef BehaviorNote BehaviorNote_t; ///< @brief Type alias for BehaviorNote

/**
 * @enum BuildPlanPolicy
 * @brief Specifies how execution plans are selected during graph building
 */
enum class BuildPlanPolicy
{
    HEURISTICS_CHOICE, ///< Use heuristics to select the best plan
    ALL ///< Build all available plans (currently unused)
};
typedef BuildPlanPolicy BuildPlanPolicy_t; ///< @brief Type alias for BuildPlanPolicy

/**
 * @enum NormFwdPhase
 * @brief Specifies the forward phase for normalization operations
 *
 * Controls whether the normalization operation computes auxiliary outputs
 * (e.g., inverse variance/RMS) needed for backward pass training.
 */
enum class NormFwdPhase
{
    NOT_SET = 0, ///< Phase not specified (invalid for execution)
    INFERENCE = 1, ///< Inference mode: only Y output computed
    TRAINING = 2 ///< Training mode: Y and inverse RMS/variance outputs computed
};
typedef NormFwdPhase NormFwdPhase_t; ///< @brief Type alias for NormFwdPhase

/**
 * @enum KnobValueType
 * @brief Specifies the data type of a knob value
 *
 * Knobs are configuration parameters for engine execution. This enum
 * indicates what type of value a particular knob expects.
 *
 * @see hipdnn_frontend::Knob
 * @see hipdnn_frontend::KnobSetting
 */
enum class KnobValueType
{
    NOT_SET = 0, ///< Value type not specified
    INT64 = 1, ///< 64-bit signed integer value
    FLOAT64 = 2, ///< 64-bit floating point value
    STRING = 3, ///< String value
};
typedef KnobValueType KnobValueType_t; ///< @brief Type alias for KnobValueType

/**
 * @brief Get the DataType enum value corresponding to a C++ type
 *
 * @tparam T The C++ type (float, half, hip_bfloat16, double, etc.)
 * @return The corresponding DataType enum value
 *
 * @code{.cpp}
 * DataType dt = getDataTypeEnumFromType<float>();  // Returns DataType::FLOAT
 * DataType dt2 = getDataTypeEnumFromType<half>();  // Returns DataType::HALF
 * @endcode
 */
template <typename T>
DataType getDataTypeEnumFromType()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return DataType::HALF;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return DataType::BFLOAT16;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return DataType::DOUBLE;
    }
    else if constexpr(std::is_same_v<T, uint8_t>)
    {
        return DataType::UINT8;
    }
    else if constexpr(std::is_same_v<T, int32_t>)
    {
        return DataType::INT32;
    }
    else if constexpr(std::is_same_v<T, int64_t>)
    {
        return DataType::INT64;
    }
    else if constexpr(std::is_same_v<T, int8_t>)
    {
        return DataType::INT8;
    }
    else if constexpr(std::is_same_v<T, fp8_e4m3>)
    {
        return DataType::FP8_E4M3;
    }
    else if constexpr(std::is_same_v<T, fp8_e4m3_fnuz>)
    {
        return DataType::FP8_E4M3_FNUZ;
    }
    else if constexpr(std::is_same_v<T, fp8_e5m2>)
    {
        return DataType::FP8_E5M2;
    }
    else if constexpr(std::is_same_v<T, fp8_e5m2_fnuz>)
    {
        return DataType::FP8_E5M2_FNUZ;
    }
    else if constexpr(std::is_same_v<T, bool>)
    {
        return DataType::BOOLEAN;
    }
    else
    {
        return DataType::NOT_SET;
    }
}

/**
 * @brief Convert frontend ConvolutionMode to backend hipdnnConvolutionMode_t
 *
 * Maps frontend convolution mode enum to the backend C API enum type for use
 * with HIPDNN_TYPE_CONVOLUTION_MODE attributes.
 *
 * @param type The frontend ConvolutionMode value
 * @return The corresponding hipdnnConvolutionMode_t value, or std::nullopt if not set
 */
inline std::optional<hipdnnConvolutionMode_t> toBackendConvMode(const ConvolutionMode& type)
{
    switch(type)
    {
    case ConvolutionMode::CROSS_CORRELATION:
        return HIPDNN_CROSS_CORRELATION;
    case ConvolutionMode::CONVOLUTION:
        return HIPDNN_CONVOLUTION;
    default:
        return std::nullopt;
    }
}

/**
 * @brief Convert backend hipdnnConvolutionMode_t to frontend ConvolutionMode
 *
 * Maps the backend C API convolution mode enum to the frontend ConvolutionMode enum.
 *
 * @param mode The backend hipdnnConvolutionMode_t value
 * @return A pair of the corresponding ConvolutionMode and an Error (set if the mode is unknown)
 */
inline std::pair<ConvolutionMode, Error> fromHipdnnConvMode(hipdnnConvolutionMode_t mode)
{
    switch(mode)
    {
    case HIPDNN_CROSS_CORRELATION:
        return {ConvolutionMode::CROSS_CORRELATION, {}};
    case HIPDNN_CONVOLUTION:
        return {ConvolutionMode::CONVOLUTION, {}};
    default:
        return {
            ConvolutionMode::NOT_SET,
            {ErrorCode::HIPDNN_BACKEND_ERROR,
             "Unknown hipdnnConvolutionMode_t value: " + std::to_string(static_cast<int>(mode))}};
    }
}

/**
 * @brief Convert frontend DiagonalAlignment to backend hipdnnDiagonalAlignment_t
 *
 * Maps frontend diagonal alignment enum directly to the backend C API enum type
 * for use with HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT attributes.
 *
 * @param type The frontend DiagonalAlignment value
 * @return The corresponding hipdnnDiagonalAlignment_t value
 */
inline hipdnnDiagonalAlignment_t toBackendDiagonalAlignment(const DiagonalAlignment& type)
{
    switch(type)
    {
    case DiagonalAlignment::TOP_LEFT:
        return HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;
    case DiagonalAlignment::BOTTOM_RIGHT:
        return HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;
    default:
        return HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;
    }
}

/**
 * @brief Convert frontend AttentionImplementation to backend hipdnnAttentionImplementation_t
 *
 * Maps frontend attention implementation enum directly to the backend C API enum type
 * for use with HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT attributes.
 *
 * @param type The frontend AttentionImplementation value
 * @return The corresponding hipdnnAttentionImplementation_t value
 */
inline hipdnnAttentionImplementation_t
    toBackendAttentionImplementation(const AttentionImplementation& type)
{
    switch(type)
    {
    case AttentionImplementation::AUTO:
        return HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT;
    case AttentionImplementation::COMPOSITE:
        return HIPDNN_ATTENTION_IMPLEMENTATION_COMPOSITE_EXT;
    case AttentionImplementation::UNIFIED:
        return HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT;
    default:
        return HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT;
    }
}

/**
 * @brief Convert backend hipdnnDiagonalAlignment_t to frontend DiagonalAlignment
 *
 * Maps the backend C API diagonal alignment enum to the frontend DiagonalAlignment enum.
 *
 * @param alignment The backend hipdnnDiagonalAlignment_t value
 * @return A pair of DiagonalAlignment and Error; error is set for unknown values
 */
inline std::pair<DiagonalAlignment, Error>
    fromHipdnnDiagonalAlignment(hipdnnDiagonalAlignment_t alignment)
{
    switch(alignment)
    {
    case HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT:
        return {DiagonalAlignment::TOP_LEFT, {}};
    case HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT:
        return {DiagonalAlignment::BOTTOM_RIGHT, {}};
    default:
        return {DiagonalAlignment::TOP_LEFT,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 "Unknown hipdnnDiagonalAlignment_t value: "
                     + std::to_string(static_cast<int>(alignment))}};
    }
}

/**
 * @brief Convert backend hipdnnAttentionImplementation_t to frontend AttentionImplementation
 *
 * Maps the backend C API attention implementation enum to the frontend
 * AttentionImplementation enum.
 *
 * @param impl The backend hipdnnAttentionImplementation_t value
 * @return A pair of AttentionImplementation and Error; error is set for unknown values
 */
inline std::pair<AttentionImplementation, Error>
    fromHipdnnAttentionImplementation(hipdnnAttentionImplementation_t impl)
{
    switch(impl)
    {
    case HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT:
        return {AttentionImplementation::AUTO, {}};
    case HIPDNN_ATTENTION_IMPLEMENTATION_COMPOSITE_EXT:
        return {AttentionImplementation::COMPOSITE, {}};
    case HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT:
        return {AttentionImplementation::UNIFIED, {}};
    default:
        return {AttentionImplementation::AUTO,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 "Unknown hipdnnAttentionImplementation_t value: "
                     + std::to_string(static_cast<int>(impl))}};
    }
}

/**
 * @brief Convert frontend NormFwdPhase to backend hipdnnNormFwdPhase_t
 *
 * Maps frontend normalization forward phase to the backend C API enum type
 * for use with HIPDNN_TYPE_NORM_FWD_PHASE attributes.
 *
 * @param type The frontend NormFwdPhase value
 * @return The corresponding hipdnnNormFwdPhase_t value, or std::nullopt if not set
 */
inline std::optional<hipdnnNormFwdPhase_t> toBackendNormFwdPhase(const NormFwdPhase& type)
{
    switch(type)
    {
    case NormFwdPhase::INFERENCE:
        return HIPDNN_NORM_FWD_INFERENCE;
    case NormFwdPhase::TRAINING:
        return HIPDNN_NORM_FWD_TRAINING;
    default:
        return std::nullopt;
    }
}

/**
 * @brief Convert backend hipdnnNormFwdPhase_t to frontend NormFwdPhase
 *
 * Maps backend C API normalization forward phase enum to the frontend enum type.
 *
 * @param phase The backend hipdnnNormFwdPhase_t value
 * @return A pair of NormFwdPhase and Error; error is set for unknown values
 */
inline std::pair<NormFwdPhase, Error> fromHipdnnNormFwdPhase(hipdnnNormFwdPhase_t phase)
{
    switch(phase)
    {
    case HIPDNN_NORM_FWD_INFERENCE:
        return {NormFwdPhase::INFERENCE, {}};
    case HIPDNN_NORM_FWD_TRAINING:
        return {NormFwdPhase::TRAINING, {}};
    default:
        return {NormFwdPhase::NOT_SET,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 "Unknown hipdnnNormFwdPhase_t value: " + std::to_string(static_cast<int>(phase))}};
    }
}

/**
 * @brief Convert frontend PointwiseMode to backend hipdnnPointwiseMode_t
 *
 * Maps frontend pointwise mode enum to the backend C API enum type for use
 * with HIPDNN_TYPE_POINTWISE_MODE attributes.
 *
 * @param type The frontend PointwiseMode value
 * @return The corresponding hipdnnPointwiseMode_t value, or std::nullopt if not set
 */
inline std::optional<hipdnnPointwiseMode_t> toBackendPointwiseMode(const PointwiseMode& type)
{
    switch(type)
    {
    case PointwiseMode::ABS:
        return HIPDNN_POINTWISE_ABS;
    case PointwiseMode::ADD:
        return HIPDNN_POINTWISE_ADD;
    case PointwiseMode::ADD_SQUARE:
        return HIPDNN_POINTWISE_ADD_SQUARE;
    case PointwiseMode::BINARY_SELECT:
        return HIPDNN_POINTWISE_BINARY_SELECT;
    case PointwiseMode::CEIL:
        return HIPDNN_POINTWISE_CEIL;
    case PointwiseMode::CMP_EQ:
        return HIPDNN_POINTWISE_CMP_EQ;
    case PointwiseMode::CMP_GE:
        return HIPDNN_POINTWISE_CMP_GE;
    case PointwiseMode::CMP_GT:
        return HIPDNN_POINTWISE_CMP_GT;
    case PointwiseMode::CMP_LE:
        return HIPDNN_POINTWISE_CMP_LE;
    case PointwiseMode::CMP_LT:
        return HIPDNN_POINTWISE_CMP_LT;
    case PointwiseMode::CMP_NEQ:
        return HIPDNN_POINTWISE_CMP_NEQ;
    case PointwiseMode::DIV:
        return HIPDNN_POINTWISE_DIV;
    case PointwiseMode::ELU_BWD:
        return HIPDNN_POINTWISE_ELU_BWD;
    case PointwiseMode::ELU_FWD:
        return HIPDNN_POINTWISE_ELU_FWD;
    case PointwiseMode::ERF:
        return HIPDNN_POINTWISE_ERF;
    case PointwiseMode::EXP:
        return HIPDNN_POINTWISE_EXP;
    case PointwiseMode::FLOOR:
        return HIPDNN_POINTWISE_FLOOR;
    case PointwiseMode::GELU_APPROX_TANH_BWD:
        return HIPDNN_POINTWISE_GELU_APPROX_TANH_BWD;
    case PointwiseMode::GELU_APPROX_TANH_FWD:
        return HIPDNN_POINTWISE_GELU_APPROX_TANH_FWD;
    case PointwiseMode::GELU_BWD:
        return HIPDNN_POINTWISE_GELU_BWD;
    case PointwiseMode::GELU_FWD:
        return HIPDNN_POINTWISE_GELU_FWD;
    case PointwiseMode::GEN_INDEX:
        return HIPDNN_POINTWISE_GEN_INDEX;
    case PointwiseMode::IDENTITY:
        return HIPDNN_POINTWISE_IDENTITY;
    case PointwiseMode::LOG:
        return HIPDNN_POINTWISE_LOG;
    case PointwiseMode::LOGICAL_AND:
        return HIPDNN_POINTWISE_LOGICAL_AND;
    case PointwiseMode::LOGICAL_NOT:
        return HIPDNN_POINTWISE_LOGICAL_NOT;
    case PointwiseMode::LOGICAL_OR:
        return HIPDNN_POINTWISE_LOGICAL_OR;
    case PointwiseMode::MAX:
        return HIPDNN_POINTWISE_MAX;
    case PointwiseMode::MIN:
        return HIPDNN_POINTWISE_MIN;
    case PointwiseMode::MUL:
        return HIPDNN_POINTWISE_MUL;
    case PointwiseMode::NEG:
        return HIPDNN_POINTWISE_NEG;
    case PointwiseMode::RECIPROCAL:
        return HIPDNN_POINTWISE_RECIPROCAL;
    case PointwiseMode::RELU_BWD:
        return HIPDNN_POINTWISE_RELU_BWD;
    case PointwiseMode::RELU_FWD:
        return HIPDNN_POINTWISE_RELU_FWD;
    case PointwiseMode::RSQRT:
        return HIPDNN_POINTWISE_RSQRT;
    case PointwiseMode::SIGMOID_BWD:
        return HIPDNN_POINTWISE_SIGMOID_BWD;
    case PointwiseMode::SIGMOID_FWD:
        return HIPDNN_POINTWISE_SIGMOID_FWD;
    case PointwiseMode::SIN:
        return HIPDNN_POINTWISE_SIN;
    case PointwiseMode::SOFTPLUS_BWD:
        return HIPDNN_POINTWISE_SOFTPLUS_BWD;
    case PointwiseMode::SOFTPLUS_FWD:
        return HIPDNN_POINTWISE_SOFTPLUS_FWD;
    case PointwiseMode::SQRT:
        return HIPDNN_POINTWISE_SQRT;
    case PointwiseMode::SUB:
        return HIPDNN_POINTWISE_SUB;
    case PointwiseMode::SWISH_BWD:
        return HIPDNN_POINTWISE_SWISH_BWD;
    case PointwiseMode::SWISH_FWD:
        return HIPDNN_POINTWISE_SWISH_FWD;
    case PointwiseMode::TAN:
        return HIPDNN_POINTWISE_TAN;
    case PointwiseMode::TANH_BWD:
        return HIPDNN_POINTWISE_TANH_BWD;
    case PointwiseMode::TANH_FWD:
        return HIPDNN_POINTWISE_TANH_FWD;
    default:
        return std::nullopt;
    }
}

/**
 * @brief Convert frontend DataType to backend hipdnnDataType_t
 *
 * @param type The frontend DataType value
 * @return The corresponding hipdnnDataType_t value, or std::nullopt if not set
 */
inline std::optional<hipdnnDataType_t> toHipdnnDataType(const DataType& type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return HIPDNN_DATA_FLOAT;
    case DataType::DOUBLE:
        return HIPDNN_DATA_DOUBLE;
    case DataType::HALF:
        return HIPDNN_DATA_HALF;
    case DataType::INT8:
        return HIPDNN_DATA_INT8;
    case DataType::INT32:
        return HIPDNN_DATA_INT32;
    case DataType::UINT8:
        return HIPDNN_DATA_UINT8;
    case DataType::BFLOAT16:
        return HIPDNN_DATA_BFLOAT16;
    case DataType::FP8_E4M3:
        return HIPDNN_DATA_FP8_E4M3;
    case DataType::FP8_E5M2:
        return HIPDNN_DATA_FP8_E5M2;
    case DataType::FP8_E8M0:
        return HIPDNN_DATA_FP8_E8M0;
    case DataType::FP4_E2M1:
        return HIPDNN_DATA_FP4_E2M1;
    case DataType::INT4:
        return HIPDNN_DATA_INT4;
    case DataType::FP6_E2M3:
        return HIPDNN_DATA_FP6_E2M3_EXT;
    case DataType::FP6_E3M2:
        return HIPDNN_DATA_FP6_E3M2_EXT;
    case DataType::INT64:
        return HIPDNN_DATA_INT64;
    case DataType::BOOLEAN:
        return HIPDNN_DATA_BOOLEAN;
    case DataType::FP8_E4M3_FNUZ:
        return HIPDNN_DATA_FP8_E4M3_FNUZ;
    case DataType::FP8_E5M2_FNUZ:
        return HIPDNN_DATA_FP8_E5M2_FNUZ;
    case DataType::NOT_SET:
    default:
        return std::nullopt;
    }
}

/**
 * @brief Convert backend hipdnnDataType_t to frontend DataType
 *
 * Maps the backend C API data type enum to the frontend DataType enum.
 *
 * @param type The backend hipdnnDataType_t value
 * @return A pair of the corresponding DataType and an Error (set if the type is unknown)
 */
inline std::pair<DataType, Error> fromHipdnnDataType(hipdnnDataType_t type)
{
    switch(type)
    {
    case HIPDNN_DATA_FLOAT:
        return {DataType::FLOAT, {}};
    case HIPDNN_DATA_DOUBLE:
        return {DataType::DOUBLE, {}};
    case HIPDNN_DATA_HALF:
        return {DataType::HALF, {}};
    case HIPDNN_DATA_INT8:
        return {DataType::INT8, {}};
    case HIPDNN_DATA_INT32:
        return {DataType::INT32, {}};
    case HIPDNN_DATA_UINT8:
        return {DataType::UINT8, {}};
    case HIPDNN_DATA_BFLOAT16:
        return {DataType::BFLOAT16, {}};
    case HIPDNN_DATA_FP8_E4M3:
        return {DataType::FP8_E4M3, {}};
    case HIPDNN_DATA_FP8_E5M2:
        return {DataType::FP8_E5M2, {}};
    case HIPDNN_DATA_FP8_E8M0:
        return {DataType::FP8_E8M0, {}};
    case HIPDNN_DATA_FP4_E2M1:
        return {DataType::FP4_E2M1, {}};
    case HIPDNN_DATA_INT4:
        return {DataType::INT4, {}};
    case HIPDNN_DATA_FP6_E2M3_EXT:
        return {DataType::FP6_E2M3, {}};
    case HIPDNN_DATA_FP6_E3M2_EXT:
        return {DataType::FP6_E3M2, {}};
    case HIPDNN_DATA_INT64:
        return {DataType::INT64, {}};
    case HIPDNN_DATA_BOOLEAN:
        return {DataType::BOOLEAN, {}};
    case HIPDNN_DATA_FP8_E4M3_FNUZ:
        return {DataType::FP8_E4M3_FNUZ, {}};
    case HIPDNN_DATA_FP8_E5M2_FNUZ:
        return {DataType::FP8_E5M2_FNUZ, {}};
    default:
        return {DataType::NOT_SET,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 "Unknown hipdnnDataType_t value: " + std::to_string(static_cast<int>(type))}};
    }
}

/// @brief Convert backend C-API hipdnnPointwiseMode_t to frontend PointwiseMode
inline std::pair<PointwiseMode, Error> fromHipdnnPointwiseMode(hipdnnPointwiseMode_t mode)
{
    switch(mode)
    {
    case HIPDNN_POINTWISE_ABS:
        return {PointwiseMode::ABS, {}};
    case HIPDNN_POINTWISE_ADD:
        return {PointwiseMode::ADD, {}};
    case HIPDNN_POINTWISE_ADD_SQUARE:
        return {PointwiseMode::ADD_SQUARE, {}};
    case HIPDNN_POINTWISE_BINARY_SELECT:
        return {PointwiseMode::BINARY_SELECT, {}};
    case HIPDNN_POINTWISE_CEIL:
        return {PointwiseMode::CEIL, {}};
    case HIPDNN_POINTWISE_CMP_EQ:
        return {PointwiseMode::CMP_EQ, {}};
    case HIPDNN_POINTWISE_CMP_GE:
        return {PointwiseMode::CMP_GE, {}};
    case HIPDNN_POINTWISE_CMP_GT:
        return {PointwiseMode::CMP_GT, {}};
    case HIPDNN_POINTWISE_CMP_LE:
        return {PointwiseMode::CMP_LE, {}};
    case HIPDNN_POINTWISE_CMP_LT:
        return {PointwiseMode::CMP_LT, {}};
    case HIPDNN_POINTWISE_CMP_NEQ:
        return {PointwiseMode::CMP_NEQ, {}};
    case HIPDNN_POINTWISE_DIV:
        return {PointwiseMode::DIV, {}};
    case HIPDNN_POINTWISE_ELU_BWD:
        return {PointwiseMode::ELU_BWD, {}};
    case HIPDNN_POINTWISE_ELU_FWD:
        return {PointwiseMode::ELU_FWD, {}};
    case HIPDNN_POINTWISE_ERF:
        return {PointwiseMode::ERF, {}};
    case HIPDNN_POINTWISE_EXP:
        return {PointwiseMode::EXP, {}};
    case HIPDNN_POINTWISE_FLOOR:
        return {PointwiseMode::FLOOR, {}};
    case HIPDNN_POINTWISE_GELU_APPROX_TANH_BWD:
        return {PointwiseMode::GELU_APPROX_TANH_BWD, {}};
    case HIPDNN_POINTWISE_GELU_APPROX_TANH_FWD:
        return {PointwiseMode::GELU_APPROX_TANH_FWD, {}};
    case HIPDNN_POINTWISE_GELU_BWD:
        return {PointwiseMode::GELU_BWD, {}};
    case HIPDNN_POINTWISE_GELU_FWD:
        return {PointwiseMode::GELU_FWD, {}};
    case HIPDNN_POINTWISE_GEN_INDEX:
        return {PointwiseMode::GEN_INDEX, {}};
    case HIPDNN_POINTWISE_IDENTITY:
        return {PointwiseMode::IDENTITY, {}};
    case HIPDNN_POINTWISE_LOG:
        return {PointwiseMode::LOG, {}};
    case HIPDNN_POINTWISE_LOGICAL_AND:
        return {PointwiseMode::LOGICAL_AND, {}};
    case HIPDNN_POINTWISE_LOGICAL_NOT:
        return {PointwiseMode::LOGICAL_NOT, {}};
    case HIPDNN_POINTWISE_LOGICAL_OR:
        return {PointwiseMode::LOGICAL_OR, {}};
    case HIPDNN_POINTWISE_MAX:
        return {PointwiseMode::MAX, {}};
    case HIPDNN_POINTWISE_MIN:
        return {PointwiseMode::MIN, {}};
    case HIPDNN_POINTWISE_MUL:
        return {PointwiseMode::MUL, {}};
    case HIPDNN_POINTWISE_NEG:
        return {PointwiseMode::NEG, {}};
    case HIPDNN_POINTWISE_RECIPROCAL:
        return {PointwiseMode::RECIPROCAL, {}};
    case HIPDNN_POINTWISE_RELU_BWD:
        return {PointwiseMode::RELU_BWD, {}};
    case HIPDNN_POINTWISE_RELU_FWD:
        return {PointwiseMode::RELU_FWD, {}};
    case HIPDNN_POINTWISE_RSQRT:
        return {PointwiseMode::RSQRT, {}};
    case HIPDNN_POINTWISE_SIGMOID_BWD:
        return {PointwiseMode::SIGMOID_BWD, {}};
    case HIPDNN_POINTWISE_SIGMOID_FWD:
        return {PointwiseMode::SIGMOID_FWD, {}};
    case HIPDNN_POINTWISE_SIN:
        return {PointwiseMode::SIN, {}};
    case HIPDNN_POINTWISE_SOFTPLUS_BWD:
        return {PointwiseMode::SOFTPLUS_BWD, {}};
    case HIPDNN_POINTWISE_SOFTPLUS_FWD:
        return {PointwiseMode::SOFTPLUS_FWD, {}};
    case HIPDNN_POINTWISE_SQRT:
        return {PointwiseMode::SQRT, {}};
    case HIPDNN_POINTWISE_SUB:
        return {PointwiseMode::SUB, {}};
    case HIPDNN_POINTWISE_SWISH_BWD:
        return {PointwiseMode::SWISH_BWD, {}};
    case HIPDNN_POINTWISE_SWISH_FWD:
        return {PointwiseMode::SWISH_FWD, {}};
    case HIPDNN_POINTWISE_TAN:
        return {PointwiseMode::TAN, {}};
    case HIPDNN_POINTWISE_TANH_BWD:
        return {PointwiseMode::TANH_BWD, {}};
    case HIPDNN_POINTWISE_TANH_FWD:
        return {PointwiseMode::TANH_FWD, {}};
    default:
        return {PointwiseMode::NOT_SET,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 "Unknown hipdnnPointwiseMode_t value: " + std::to_string(static_cast<int>(mode))}};
    }
}

/// @brief Convert frontend HeuristicMode to backend heuristic mode
inline hipdnnBackendHeurMode_t toBackendType(const HeuristicMode& type)
{
    switch(type)
    {
    // All cuDNN heuristic modes currently fold to hipDNN fallback.
    case HeuristicMode::FALLBACK:
    case HeuristicMode::A:
    case HeuristicMode::B:
    case HeuristicMode::OPENSOURCE:
    default:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    }
}

/// @brief Convert backend behavior note to frontend behavior note.
/// @return A frontend behavior note. Unknown values are preserved numerically.
inline BehaviorNote fromHipdnnBehaviorNote(hipdnnBackendBehaviorNote_t note)
{
    switch(note)
    {
    case HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
        return BehaviorNote::RUNTIME_COMPILATION;
    case HIPDNN_BEHAVIOR_NOTE_REQUIRES_LAYOUT_TRANSFORM:
        return BehaviorNote::REQUIRES_LAYOUT_TRANSFORM;
    case HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE:
        return BehaviorNote::SUPPORTS_GRAPH_CAPTURE;
    case HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY:
        return BehaviorNote::EXTERNAL_LIBRARY_DEPENDENCY;
    case HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION:
        return BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION;
    default:
        return static_cast<BehaviorNote>(note);
    }
}

/// @brief Return true if a behavior note is known to this frontend version.
inline bool isKnownBehaviorNote(const BehaviorNote& note)
{
    switch(note)
    {
    case BehaviorNote::RUNTIME_COMPILATION:
    case BehaviorNote::REQUIRES_LAYOUT_TRANSFORM:
    case BehaviorNote::SUPPORTS_GRAPH_CAPTURE:
    case BehaviorNote::EXTERNAL_LIBRARY_DEPENDENCY:
    case BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION:
        return true;
    default:
        return false;
    }
}

/// @brief Convert BehaviorNote to a human-readable string
/// @param note The behavior note to convert
/// @return A C-string representation of the behavior note
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const BehaviorNote& note)
{
    switch(note)
    {
    case BehaviorNote::RUNTIME_COMPILATION:
        return "RUNTIME_COMPILATION";
    case BehaviorNote::REQUIRES_LAYOUT_TRANSFORM:
        return "REQUIRES_LAYOUT_TRANSFORM";
    case BehaviorNote::SUPPORTS_GRAPH_CAPTURE:
        return "SUPPORTS_GRAPH_CAPTURE";
    case BehaviorNote::EXTERNAL_LIBRARY_DEPENDENCY:
        return "EXTERNAL_LIBRARY_DEPENDENCY";
    case BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION:
        return "SUPPORTS_EXECUTION_PLAN_SERIALIZATION";
    default:
        return "unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, const BehaviorNote& note)
{
    os << to_string(note);
    return os;
}

/**
 * @brief Convert ConvolutionMode to a human-readable string
 * @param mode The convolution mode to convert
 * @return A C-string representation of the convolution mode
 */
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const ConvolutionMode& mode)
{
    switch(mode)
    {
    case ConvolutionMode::NOT_SET:
        return "not_set";
    case ConvolutionMode::CROSS_CORRELATION:
        return "cross_correlation";
    case ConvolutionMode::CONVOLUTION:
        return "convolution";
    default:
        return "unknown";
    }
}

/// @brief Stream insertion operator for ConvolutionMode
inline std::ostream& operator<<(std::ostream& os, const ConvolutionMode& mode)
{
    return os << to_string(mode);
}

/// @brief Get a human-readable string for a DataType value
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const DataType& type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return "fp32";
    case DataType::HALF:
        return "fp16";
    case DataType::BFLOAT16:
        return "bf16";
    case DataType::DOUBLE:
        return "fp64";
    case DataType::UINT8:
        return "uint8";
    case DataType::INT32:
        return "int32";
    case DataType::INT8:
        return "int8";
    case DataType::FP8_E4M3:
        return "fp8_e4m3";
    case DataType::FP8_E5M2:
        return "fp8_e5m2";
    case DataType::FP8_E8M0:
        return "fp8_e8m0";
    case DataType::FP4_E2M1:
        return "fp4_e2m1";
    case DataType::INT4:
        return "int4";
    case DataType::FP6_E2M3:
        return "fp6_e2m3";
    case DataType::FP6_E3M2:
        return "fp6_e3m2";
    case DataType::INT64:
        return "int64";
    case DataType::BOOLEAN:
        return "boolean";
    case DataType::FP8_E4M3_FNUZ:
        return "fp8_e4m3_fnuz";
    case DataType::FP8_E5M2_FNUZ:
        return "fp8_e5m2_fnuz";
    case DataType::INT8x4:
        return "int8x4";
    case DataType::UINT8x4:
        return "uint8x4";
    case DataType::INT8x32:
        return "int8x32";
    case DataType::FAST_FLOAT_FOR_FP8:
        return "fast_float_for_fp8";
    case DataType::COMPLEX_FP32:
        return "complex_fp32";
    case DataType::COMPLEX_FP64:
        return "complex_fp64";
    default:
        return "unknown";
    }
}

/// @brief Stream insertion operator for DataType
inline std::ostream& operator<<(std::ostream& os, const DataType& type)
{
    return os << to_string(type);
}

/// @brief Get a human-readable string for a PointwiseMode value
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const PointwiseMode& mode)
{
    switch(mode)
    {
    case PointwiseMode::NOT_SET:
        return "NOT_SET";
    case PointwiseMode::ABS:
        return "ABS";
    case PointwiseMode::ADD:
        return "ADD";
    case PointwiseMode::ADD_SQUARE:
        return "ADD_SQUARE";
    case PointwiseMode::BINARY_SELECT:
        return "BINARY_SELECT";
    case PointwiseMode::CEIL:
        return "CEIL";
    case PointwiseMode::CMP_EQ:
        return "CMP_EQ";
    case PointwiseMode::CMP_GE:
        return "CMP_GE";
    case PointwiseMode::CMP_GT:
        return "CMP_GT";
    case PointwiseMode::CMP_LE:
        return "CMP_LE";
    case PointwiseMode::CMP_LT:
        return "CMP_LT";
    case PointwiseMode::CMP_NEQ:
        return "CMP_NEQ";
    case PointwiseMode::DIV:
        return "DIV";
    case PointwiseMode::ELU_BWD:
        return "ELU_BWD";
    case PointwiseMode::ELU_FWD:
        return "ELU_FWD";
    case PointwiseMode::ERF:
        return "ERF";
    case PointwiseMode::EXP:
        return "EXP";
    case PointwiseMode::FLOOR:
        return "FLOOR";
    case PointwiseMode::GELU_APPROX_TANH_BWD:
        return "GELU_APPROX_TANH_BWD";
    case PointwiseMode::GELU_APPROX_TANH_FWD:
        return "GELU_APPROX_TANH_FWD";
    case PointwiseMode::GELU_BWD:
        return "GELU_BWD";
    case PointwiseMode::GELU_FWD:
        return "GELU_FWD";
    case PointwiseMode::GEN_INDEX:
        return "GEN_INDEX";
    case PointwiseMode::IDENTITY:
        return "IDENTITY";
    case PointwiseMode::LOG:
        return "LOG";
    case PointwiseMode::LOGICAL_AND:
        return "LOGICAL_AND";
    case PointwiseMode::LOGICAL_NOT:
        return "LOGICAL_NOT";
    case PointwiseMode::LOGICAL_OR:
        return "LOGICAL_OR";
    case PointwiseMode::MAX:
        return "MAX";
    case PointwiseMode::MIN:
        return "MIN";
    case PointwiseMode::MUL:
        return "MUL";
    case PointwiseMode::NEG:
        return "NEG";
    case PointwiseMode::RECIPROCAL:
        return "RECIPROCAL";
    case PointwiseMode::RELU_BWD:
        return "RELU_BWD";
    case PointwiseMode::RELU_FWD:
        return "RELU_FWD";
    case PointwiseMode::RSQRT:
        return "RSQRT";
    case PointwiseMode::SIGMOID_BWD:
        return "SIGMOID_BWD";
    case PointwiseMode::SIGMOID_FWD:
        return "SIGMOID_FWD";
    case PointwiseMode::SIN:
        return "SIN";
    case PointwiseMode::SOFTPLUS_BWD:
        return "SOFTPLUS_BWD";
    case PointwiseMode::SOFTPLUS_FWD:
        return "SOFTPLUS_FWD";
    case PointwiseMode::SQRT:
        return "SQRT";
    case PointwiseMode::SUB:
        return "SUB";
    case PointwiseMode::SWISH_BWD:
        return "SWISH_BWD";
    case PointwiseMode::SWISH_FWD:
        return "SWISH_FWD";
    case PointwiseMode::TAN:
        return "TAN";
    case PointwiseMode::TANH_BWD:
        return "TANH_BWD";
    case PointwiseMode::TANH_FWD:
        return "TANH_FWD";
    case PointwiseMode::MOD:
        return "MOD";
    case PointwiseMode::POW:
        return "POW";
    case PointwiseMode::COS:
        return "COS";
    default:
        return "UNKNOWN";
    }
}

/// @brief Stream insertion operator for PointwiseMode
inline std::ostream& operator<<(std::ostream& os, const PointwiseMode& mode)
{
    return os << to_string(mode);
}

/// @brief Get a human-readable string for a BuildPlanPolicy value
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const BuildPlanPolicy& policy)
{
    switch(policy)
    {
    case BuildPlanPolicy::HEURISTICS_CHOICE:
        return "HEURISTICS_CHOICE";
    case BuildPlanPolicy::ALL:
        return "ALL";
    default:
        return "unknown";
    }
}

/// @brief Stream insertion operator for BuildPlanPolicy
inline std::ostream& operator<<(std::ostream& os, const BuildPlanPolicy& policy)
{
    return os << to_string(policy);
}

/// @brief Get a human-readable string for a HeuristicMode value
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const HeuristicMode& mode)
{
    switch(mode)
    {
    case HeuristicMode::FALLBACK:
        return "FALLBACK";
    case HeuristicMode::A:
        return "A";
    case HeuristicMode::B:
        return "B";
    case HeuristicMode::OPENSOURCE:
        return "OPENSOURCE";
    default:
        return "unknown";
    }
}

/// @brief Stream insertion operator for HeuristicMode
inline std::ostream& operator<<(std::ostream& os, const HeuristicMode& mode)
{
    return os << to_string(mode);
}

/// @brief Convert frontend ReductionMode to backend C-API hipdnnReduceTensorOp_t
inline std::optional<hipdnnReduceTensorOp_t> toBackendReductionMode(const ReductionMode& type)
{
    switch(type)
    {
    case ReductionMode::ADD:
        return HIPDNN_REDUCE_TENSOR_ADD;
    case ReductionMode::MUL:
        return HIPDNN_REDUCE_TENSOR_MUL;
    case ReductionMode::MIN:
        return HIPDNN_REDUCE_TENSOR_MIN;
    case ReductionMode::MAX:
        return HIPDNN_REDUCE_TENSOR_MAX;
    case ReductionMode::AMAX:
        return HIPDNN_REDUCE_TENSOR_AMAX;
    case ReductionMode::AVG:
        return HIPDNN_REDUCE_TENSOR_AVG;
    case ReductionMode::NORM1:
        return HIPDNN_REDUCE_TENSOR_NORM1;
    case ReductionMode::NORM2:
        return HIPDNN_REDUCE_TENSOR_NORM2;
    case ReductionMode::MUL_NO_ZEROS:
        return HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS;
    default:
        return std::nullopt;
    }
}

/// @brief Convert backend C-API hipdnnReduceTensorOp_t to frontend ReductionMode
inline std::pair<ReductionMode, Error> fromHipdnnReduceTensorOp(hipdnnReduceTensorOp_t mode)
{
    switch(mode)
    {
    case HIPDNN_REDUCE_TENSOR_ADD:
        return {ReductionMode::ADD, {}};
    case HIPDNN_REDUCE_TENSOR_MUL:
        return {ReductionMode::MUL, {}};
    case HIPDNN_REDUCE_TENSOR_MIN:
        return {ReductionMode::MIN, {}};
    case HIPDNN_REDUCE_TENSOR_MAX:
        return {ReductionMode::MAX, {}};
    case HIPDNN_REDUCE_TENSOR_AMAX:
        return {ReductionMode::AMAX, {}};
    case HIPDNN_REDUCE_TENSOR_AVG:
        return {ReductionMode::AVG, {}};
    case HIPDNN_REDUCE_TENSOR_NORM1:
        return {ReductionMode::NORM1, {}};
    case HIPDNN_REDUCE_TENSOR_NORM2:
        return {ReductionMode::NORM2, {}};
    case HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
        return {ReductionMode::MUL_NO_ZEROS, {}};
    default:
        return {
            ReductionMode::NOT_SET,
            {ErrorCode::HIPDNN_BACKEND_ERROR,
             "Unknown hipdnnReduceTensorOp_t value: " + std::to_string(static_cast<int>(mode))}};
    }
}

/// @brief Get a human-readable string for a ReductionMode value
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const ReductionMode& mode)
{
    switch(mode)
    {
    case ReductionMode::NOT_SET:
        return "NOT_SET";
    case ReductionMode::ADD:
        return "ADD";
    case ReductionMode::MUL:
        return "MUL";
    case ReductionMode::MIN:
        return "MIN";
    case ReductionMode::MAX:
        return "MAX";
    case ReductionMode::AMAX:
        return "AMAX";
    case ReductionMode::AVG:
        return "AVG";
    case ReductionMode::NORM1:
        return "NORM1";
    case ReductionMode::NORM2:
        return "NORM2";
    case ReductionMode::MUL_NO_ZEROS:
        return "MUL_NO_ZEROS";
    default:
        return "UNKNOWN";
    }
}

/// @brief Stream insertion operator for ReductionMode
inline std::ostream& operator<<(std::ostream& os, const ReductionMode& mode)
{
    return os << to_string(mode);
}

/// @brief Get a human-readable string for a KnobValueType value
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const KnobValueType& type)
{
    switch(type)
    {
    case KnobValueType::INT64:
        return "int64";
    case KnobValueType::FLOAT64:
        return "float64";
    case KnobValueType::STRING:
        return "string";
    default:
        return "unknown";
    }
}

/// @brief Stream insertion operator for KnobValueType
inline std::ostream& operator<<(std::ostream& os, const KnobValueType& type)
{
    return os << to_string(type);
}

/// @brief Get a human-readable string for a NormFwdPhase value
// NOLINTNEXTLINE(readability-identifier-naming)
inline const char* to_string(const NormFwdPhase& phase)
{
    switch(phase)
    {
    case NormFwdPhase::INFERENCE:
        return "INFERENCE";
    case NormFwdPhase::TRAINING:
        return "TRAINING";
    default:
        return "NOT_SET";
    }
}

/// @brief Stream insertion operator for NormFwdPhase
inline std::ostream& operator<<(std::ostream& os, const NormFwdPhase& phase)
{
    return os << to_string(phase);
}

/**
 * @brief Determine the KnobValueType from a variant's active alternative
 * @tparam Ts Variant alternative types
 * @param value The variant to inspect
 * @return The KnobValueType matching the currently held type
 */
template <typename... Ts>
inline KnobValueType getKnobValueTypeFromVariant(const std::variant<Ts...>& value)
{
    KnobValueType ret = KnobValueType::INT64;

    if(std::holds_alternative<int64_t>(value))
    {
        ret = KnobValueType::INT64;
    }
    else if(std::holds_alternative<double>(value))
    {
        ret = KnobValueType::FLOAT64;
    }
    else if(std::holds_alternative<std::string>(value))
    {
        ret = KnobValueType::STRING;
    }

    return ret;
}

// NOTE: Parallel mode classification exists in data_sdk PointwiseValidation.hpp.
// Keep both in sync when adding new PointwiseMode values.

/**
 * @brief Check if a pointwise mode is a unary operation (1 input)
 * @param mode The pointwise mode to check
 * @return true if the mode is unary, false otherwise
 */
inline bool isUnaryPointwiseMode(PointwiseMode mode)
{
    switch(mode)
    {
    case PointwiseMode::ABS:
    case PointwiseMode::CEIL:
    case PointwiseMode::ELU_FWD:
    case PointwiseMode::ERF:
    case PointwiseMode::EXP:
    case PointwiseMode::FLOOR:
    case PointwiseMode::GELU_APPROX_TANH_FWD:
    case PointwiseMode::GELU_FWD:
    case PointwiseMode::GEN_INDEX:
    case PointwiseMode::IDENTITY:
    case PointwiseMode::LOG:
    case PointwiseMode::LOGICAL_NOT:
    case PointwiseMode::NEG:
    case PointwiseMode::RECIPROCAL:
    case PointwiseMode::RELU_FWD:
    case PointwiseMode::RSQRT:
    case PointwiseMode::SIGMOID_FWD:
    case PointwiseMode::SIN:
    case PointwiseMode::SOFTPLUS_FWD:
    case PointwiseMode::SQRT:
    case PointwiseMode::SWISH_FWD:
    case PointwiseMode::TAN:
    case PointwiseMode::TANH_FWD:
    case PointwiseMode::COS:
        return true;
    default:
        return false;
    }
}

/**
 * @brief Check if a pointwise mode is a binary operation (2 inputs)
 * @param mode The pointwise mode to check
 * @return true if the mode is binary, false otherwise
 */
inline bool isBinaryPointwiseMode(PointwiseMode mode)
{
    switch(mode)
    {
    case PointwiseMode::ADD:
    case PointwiseMode::ADD_SQUARE:
    case PointwiseMode::CMP_EQ:
    case PointwiseMode::CMP_GE:
    case PointwiseMode::CMP_GT:
    case PointwiseMode::CMP_LE:
    case PointwiseMode::CMP_LT:
    case PointwiseMode::CMP_NEQ:
    case PointwiseMode::DIV:
    case PointwiseMode::ELU_BWD:
    case PointwiseMode::GELU_APPROX_TANH_BWD:
    case PointwiseMode::GELU_BWD:
    case PointwiseMode::LOGICAL_AND:
    case PointwiseMode::LOGICAL_OR:
    case PointwiseMode::MAX:
    case PointwiseMode::MIN:
    case PointwiseMode::MUL:
    case PointwiseMode::RELU_BWD:
    case PointwiseMode::SIGMOID_BWD:
    case PointwiseMode::SOFTPLUS_BWD:
    case PointwiseMode::SUB:
    case PointwiseMode::SWISH_BWD:
    case PointwiseMode::TANH_BWD:
    case PointwiseMode::MOD:
    case PointwiseMode::POW:
        return true;
    default:
        return false;
    }
}

/**
 * @brief Check if a pointwise mode is a ternary operation (3 inputs)
 * @param mode The pointwise mode to check
 * @return true if the mode is ternary, false otherwise
 */
inline bool isTernaryPointwiseMode(PointwiseMode mode)
{
    return mode == PointwiseMode::BINARY_SELECT;
}

/**
 * @brief Convert frontend ResampleMode to backend hipdnnResampleMode_t
 */
inline std::optional<hipdnnResampleMode_t> toBackendResampleMode(const ResampleMode& type)
{
    switch(type)
    {
    case ResampleMode::MAXPOOL:
        return HIPDNN_RESAMPLE_MAXPOOL;
    case ResampleMode::AVGPOOL_EXCLUDE_PADDING:
        return HIPDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;
    case ResampleMode::AVGPOOL_INCLUDE_PADDING:
        return HIPDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
    default:
        return std::nullopt;
    }
}

/**
 * @brief Convert backend hipdnnResampleMode_t to frontend ResampleMode
 */
inline std::pair<ResampleMode, Error> fromHipdnnResampleMode(hipdnnResampleMode_t mode)
{
    switch(mode)
    {
    case HIPDNN_RESAMPLE_MAXPOOL:
        return {ResampleMode::MAXPOOL, {}};
    case HIPDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING:
        return {ResampleMode::AVGPOOL_EXCLUDE_PADDING, {}};
    case HIPDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING:
        return {ResampleMode::AVGPOOL_INCLUDE_PADDING, {}};
    default:
        return {ResampleMode::NOT_SET,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 "Unknown hipdnnResampleMode_t value: " + std::to_string(static_cast<int>(mode))}};
    }
}

/**
 * @brief Convert frontend PaddingMode to backend hipdnnPaddingMode_t
 */
inline std::optional<hipdnnPaddingMode_t> toBackendPaddingMode(const PaddingMode& type)
{
    switch(type)
    {
    case PaddingMode::NEG_INF_PAD:
        return HIPDNN_PADDING_NEG_INF_PAD;
    case PaddingMode::ZERO_PAD:
        return HIPDNN_PADDING_ZERO_PAD;
    default:
        return std::nullopt;
    }
}

/**
 * @brief Convert backend hipdnnPaddingMode_t to frontend PaddingMode
 */
inline std::pair<PaddingMode, Error> fromHipdnnPaddingMode(hipdnnPaddingMode_t mode)
{
    switch(mode)
    {
    case HIPDNN_PADDING_NEG_INF_PAD:
        return {PaddingMode::NEG_INF_PAD, {}};
    case HIPDNN_PADDING_ZERO_PAD:
        return {PaddingMode::ZERO_PAD, {}};
    default:
        return {PaddingMode::NOT_SET,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 "Unknown hipdnnPaddingMode_t value: " + std::to_string(static_cast<int>(mode))}};
    }
}

} // namespace hipdnn_frontend

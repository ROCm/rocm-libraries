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
 * the internal SDK/backend types.
 */

#pragma once

#include <HipdnnAttentionImplementation.h>
#include <HipdnnBackendHeuristicType.h>
#include <HipdnnConvolutionMode.h>
#include <HipdnnDataType.h>
#include <HipdnnDiagonalAlignment.h>
#include <HipdnnNormFwdPhase.h>
#include <HipdnnPointwiseMode.h>
#include <hipdnn_data_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/data_objects/norm_common_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PointwiseValidation.hpp>

#include <hipdnn_frontend/Error.hpp>

#include <bitset>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <variant>

namespace hipdnn_frontend
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_data_sdk::types::fp8_e5m2;
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
    ADD_SQUARE = 3, ///< Add and square: (x + y)²
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
};
typedef PointwiseMode PointwiseMode_t; ///< @brief Type alias for PointwiseMode

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
 * @see SdpaAttributes::set_diagonal_alignment()
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
 * @see SdpaAttributes::set_implementation()
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
};
typedef HeuristicMode HeurMode_t; ///< @brief Type alias for HeuristicMode

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
 * @see Knob, KnobSetting
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
    else if constexpr(std::is_same_v<T, int8_t>)
    {
        return DataType::INT8;
    }
    else if constexpr(std::is_same_v<T, fp8_e4m3>)
    {
        return DataType::FP8_E4M3;
    }
    else if constexpr(std::is_same_v<T, fp8_e5m2>)
    {
        return DataType::FP8_E5M2;
    }
    else
    {
        return DataType::NOT_SET;
    }
}

/// @brief Convert frontend ConvolutionMode to SDK ConvMode
inline hipdnn_data_sdk::data_objects::ConvMode toSdkType(const ConvolutionMode& type)
{
    switch(type)
    {
    case ConvolutionMode::CROSS_CORRELATION:
        return hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    case ConvolutionMode::CONVOLUTION:
        return hipdnn_data_sdk::data_objects::ConvMode::CONVOLUTION;
    default:
        return hipdnn_data_sdk::data_objects::ConvMode::UNSET;
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
        return HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;
    case ConvolutionMode::CONVOLUTION:
        return HIPDNN_CONVOLUTION_MODE_CONVOLUTION;
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
    case HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION:
        return {ConvolutionMode::CROSS_CORRELATION, {}};
    case HIPDNN_CONVOLUTION_MODE_CONVOLUTION:
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
 * for use with HIPDNN_TYPE_DIAGONAL_ALIGNMENT attributes.
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
 * for use with HIPDNN_TYPE_ATTENTION_IMPLEMENTATION attributes.
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
        return HIPDNN_NORM_FWD_PHASE_INFERENCE;
    case NormFwdPhase::TRAINING:
        return HIPDNN_NORM_FWD_PHASE_TRAINING;
    default:
        return std::nullopt;
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

/// @brief Convert SDK ConvMode to frontend ConvolutionMode
inline hipdnn_frontend::ConvolutionMode
    fromSdkType(const hipdnn_data_sdk::data_objects::ConvMode& type)
{
    switch(type)
    {
    case hipdnn_data_sdk::data_objects::ConvMode::CROSS_CORRELATION:
        return hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION;
    case hipdnn_data_sdk::data_objects::ConvMode::CONVOLUTION:
        return hipdnn_frontend::ConvolutionMode::CONVOLUTION;
    default:
        return hipdnn_frontend::ConvolutionMode::NOT_SET;
    }
}

/// @brief Convert frontend DataType to SDK DataType
inline hipdnn_data_sdk::data_objects::DataType toSdkType(const DataType& type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return hipdnn_data_sdk::data_objects::DataType::FLOAT;
    case DataType::HALF:
        return hipdnn_data_sdk::data_objects::DataType::HALF;
    case DataType::BFLOAT16:
        return hipdnn_data_sdk::data_objects::DataType::BFLOAT16;
    case DataType::DOUBLE:
        return hipdnn_data_sdk::data_objects::DataType::DOUBLE;
    case DataType::UINT8:
        return hipdnn_data_sdk::data_objects::DataType::UINT8;
    case DataType::INT32:
        return hipdnn_data_sdk::data_objects::DataType::INT32;
    case DataType::INT8:
        return hipdnn_data_sdk::data_objects::DataType::INT8;
    case DataType::FP8_E4M3:
        return hipdnn_data_sdk::data_objects::DataType::FP8_E4M3;
    case DataType::FP8_E5M2:
        return hipdnn_data_sdk::data_objects::DataType::FP8_E5M2;
    default:
        return hipdnn_data_sdk::data_objects::DataType::UNSET;
    }
}

/// @brief Convert SDK DataType to frontend DataType
inline hipdnn_frontend::DataType fromSdkType(const hipdnn_data_sdk::data_objects::DataType& type)
{
    switch(type)
    {
    case hipdnn_data_sdk::data_objects::DataType::FLOAT:
        return hipdnn_frontend::DataType::FLOAT;
    case hipdnn_data_sdk::data_objects::DataType::HALF:
        return hipdnn_frontend::DataType::HALF;
    case hipdnn_data_sdk::data_objects::DataType::BFLOAT16:
        return hipdnn_frontend::DataType::BFLOAT16;
    case hipdnn_data_sdk::data_objects::DataType::DOUBLE:
        return hipdnn_frontend::DataType::DOUBLE;
    case hipdnn_data_sdk::data_objects::DataType::UINT8:
        return hipdnn_frontend::DataType::UINT8;
    case hipdnn_data_sdk::data_objects::DataType::INT32:
        return hipdnn_frontend::DataType::INT32;
    case hipdnn_data_sdk::data_objects::DataType::INT8:
        return hipdnn_frontend::DataType::INT8;
    case hipdnn_data_sdk::data_objects::DataType::FP8_E4M3:
        return hipdnn_frontend::DataType::FP8_E4M3;
    case hipdnn_data_sdk::data_objects::DataType::FP8_E5M2:
        return hipdnn_frontend::DataType::FP8_E5M2;
    default:
        return hipdnn_frontend::DataType::NOT_SET;
    }
}

/// @brief Convert frontend DiagonalAlignment to SDK DiagonalAlignment
inline hipdnn_data_sdk::data_objects::DiagonalAlignment toSdkType(const DiagonalAlignment& type)
{
    switch(type)
    {
    case DiagonalAlignment::TOP_LEFT:
        return hipdnn_data_sdk::data_objects::DiagonalAlignment::TOP_LEFT;
    case DiagonalAlignment::BOTTOM_RIGHT:
        return hipdnn_data_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    default:
        return hipdnn_data_sdk::data_objects::DiagonalAlignment::TOP_LEFT;
    }
}

/// @brief Convert SDK DiagonalAlignment to frontend DiagonalAlignment
inline hipdnn_frontend::DiagonalAlignment
    fromSdkType(const hipdnn_data_sdk::data_objects::DiagonalAlignment& type)
{
    switch(type)
    {
    case hipdnn_data_sdk::data_objects::DiagonalAlignment::TOP_LEFT:
        return hipdnn_frontend::DiagonalAlignment::TOP_LEFT;
    case hipdnn_data_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT:
        return hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT;
    default:
        return hipdnn_frontend::DiagonalAlignment::TOP_LEFT;
    }
}

/// @brief Convert frontend AttentionImplementation to SDK AttentionImplementation
inline hipdnn_data_sdk::data_objects::AttentionImplementation
    toSdkType(const AttentionImplementation& type)
{
    switch(type)
    {
    case AttentionImplementation::AUTO:
        return hipdnn_data_sdk::data_objects::AttentionImplementation::AUTO;
    case AttentionImplementation::COMPOSITE:
        return hipdnn_data_sdk::data_objects::AttentionImplementation::COMPOSITE;
    case AttentionImplementation::UNIFIED:
        return hipdnn_data_sdk::data_objects::AttentionImplementation::UNIFIED;
    default:
        return hipdnn_data_sdk::data_objects::AttentionImplementation::AUTO;
    }
}

/// @brief Convert SDK AttentionImplementation to frontend AttentionImplementation
inline hipdnn_frontend::AttentionImplementation
    fromSdkType(const hipdnn_data_sdk::data_objects::AttentionImplementation& type)
{
    switch(type)
    {
    case hipdnn_data_sdk::data_objects::AttentionImplementation::AUTO:
        return hipdnn_frontend::AttentionImplementation::AUTO;
    case hipdnn_data_sdk::data_objects::AttentionImplementation::COMPOSITE:
        return hipdnn_frontend::AttentionImplementation::COMPOSITE;
    case hipdnn_data_sdk::data_objects::AttentionImplementation::UNIFIED:
        return hipdnn_frontend::AttentionImplementation::UNIFIED;
    default:
        return hipdnn_frontend::AttentionImplementation::AUTO;
    }
}

/// @brief Convert frontend DataType to backend hipdnnDataType_t
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
    default:
        return {DataType::NOT_SET,
                {ErrorCode::HIPDNN_BACKEND_ERROR,
                 "Unknown hipdnnDataType_t value: " + std::to_string(static_cast<int>(type))}};
    }
}

/// @brief Convert frontend PointwiseMode to SDK PointwiseMode
inline hipdnn_data_sdk::data_objects::PointwiseMode toSdkType(const PointwiseMode& type)
{
    switch(type)
    {
    case PointwiseMode::ABS:
        return hipdnn_data_sdk::data_objects::PointwiseMode::ABS;
    case PointwiseMode::ADD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::ADD;
    case PointwiseMode::ADD_SQUARE:
        return hipdnn_data_sdk::data_objects::PointwiseMode::ADD_SQUARE;
    case PointwiseMode::BINARY_SELECT:
        return hipdnn_data_sdk::data_objects::PointwiseMode::BINARY_SELECT;
    case PointwiseMode::CEIL:
        return hipdnn_data_sdk::data_objects::PointwiseMode::CEIL;
    case PointwiseMode::CMP_EQ:
        return hipdnn_data_sdk::data_objects::PointwiseMode::CMP_EQ;
    case PointwiseMode::CMP_GE:
        return hipdnn_data_sdk::data_objects::PointwiseMode::CMP_GE;
    case PointwiseMode::CMP_GT:
        return hipdnn_data_sdk::data_objects::PointwiseMode::CMP_GT;
    case PointwiseMode::CMP_LE:
        return hipdnn_data_sdk::data_objects::PointwiseMode::CMP_LE;
    case PointwiseMode::CMP_LT:
        return hipdnn_data_sdk::data_objects::PointwiseMode::CMP_LT;
    case PointwiseMode::CMP_NEQ:
        return hipdnn_data_sdk::data_objects::PointwiseMode::CMP_NEQ;
    case PointwiseMode::DIV:
        return hipdnn_data_sdk::data_objects::PointwiseMode::DIV;
    case PointwiseMode::ELU_BWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::ELU_BWD;
    case PointwiseMode::ELU_FWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::ELU_FWD;
    case PointwiseMode::ERF:
        return hipdnn_data_sdk::data_objects::PointwiseMode::ERF;
    case PointwiseMode::EXP:
        return hipdnn_data_sdk::data_objects::PointwiseMode::EXP;
    case PointwiseMode::FLOOR:
        return hipdnn_data_sdk::data_objects::PointwiseMode::FLOOR;
    case PointwiseMode::GELU_APPROX_TANH_BWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_BWD;
    case PointwiseMode::GELU_APPROX_TANH_FWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD;
    case PointwiseMode::GELU_BWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::GELU_BWD;
    case PointwiseMode::GELU_FWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::GELU_FWD;
    case PointwiseMode::GEN_INDEX:
        return hipdnn_data_sdk::data_objects::PointwiseMode::GEN_INDEX;
    case PointwiseMode::IDENTITY:
        return hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY;
    case PointwiseMode::LOG:
        return hipdnn_data_sdk::data_objects::PointwiseMode::LOG;
    case PointwiseMode::LOGICAL_AND:
        return hipdnn_data_sdk::data_objects::PointwiseMode::LOGICAL_AND;
    case PointwiseMode::LOGICAL_NOT:
        return hipdnn_data_sdk::data_objects::PointwiseMode::LOGICAL_NOT;
    case PointwiseMode::LOGICAL_OR:
        return hipdnn_data_sdk::data_objects::PointwiseMode::LOGICAL_OR;
    case PointwiseMode::MAX:
        return hipdnn_data_sdk::data_objects::PointwiseMode::MAX_OP;
    case PointwiseMode::MIN:
        return hipdnn_data_sdk::data_objects::PointwiseMode::MIN_OP;
    case PointwiseMode::MUL:
        return hipdnn_data_sdk::data_objects::PointwiseMode::MUL;
    case PointwiseMode::NEG:
        return hipdnn_data_sdk::data_objects::PointwiseMode::NEG;
    case PointwiseMode::RECIPROCAL:
        return hipdnn_data_sdk::data_objects::PointwiseMode::RECIPROCAL;
    case PointwiseMode::RELU_BWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD;
    case PointwiseMode::RELU_FWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD;
    case PointwiseMode::RSQRT:
        return hipdnn_data_sdk::data_objects::PointwiseMode::RSQRT;
    case PointwiseMode::SIGMOID_BWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_BWD;
    case PointwiseMode::SIGMOID_FWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_FWD;
    case PointwiseMode::SIN:
        return hipdnn_data_sdk::data_objects::PointwiseMode::SIN;
    case PointwiseMode::SOFTPLUS_BWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD;
    case PointwiseMode::SOFTPLUS_FWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD;
    case PointwiseMode::SQRT:
        return hipdnn_data_sdk::data_objects::PointwiseMode::SQRT;
    case PointwiseMode::SUB:
        return hipdnn_data_sdk::data_objects::PointwiseMode::SUB;
    case PointwiseMode::SWISH_BWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::SWISH_BWD;
    case PointwiseMode::SWISH_FWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::SWISH_FWD;
    case PointwiseMode::TAN:
        return hipdnn_data_sdk::data_objects::PointwiseMode::TAN;
    case PointwiseMode::TANH_BWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::TANH_BWD;
    case PointwiseMode::TANH_FWD:
        return hipdnn_data_sdk::data_objects::PointwiseMode::TANH_FWD;
    default:
        return hipdnn_data_sdk::data_objects::PointwiseMode::UNSET;
    }
}

/// @brief Convert SDK PointwiseMode to frontend PointwiseMode
inline hipdnn_frontend::PointwiseMode
    fromSdkType(const hipdnn_data_sdk::data_objects::PointwiseMode& type)
{
    switch(type)
    {
    case hipdnn_data_sdk::data_objects::PointwiseMode::ABS:
        return hipdnn_frontend::PointwiseMode::ABS;
    case hipdnn_data_sdk::data_objects::PointwiseMode::ADD:
        return hipdnn_frontend::PointwiseMode::ADD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::ADD_SQUARE:
        return hipdnn_frontend::PointwiseMode::ADD_SQUARE;
    case hipdnn_data_sdk::data_objects::PointwiseMode::BINARY_SELECT:
        return hipdnn_frontend::PointwiseMode::BINARY_SELECT;
    case hipdnn_data_sdk::data_objects::PointwiseMode::CEIL:
        return hipdnn_frontend::PointwiseMode::CEIL;
    case hipdnn_data_sdk::data_objects::PointwiseMode::CMP_EQ:
        return hipdnn_frontend::PointwiseMode::CMP_EQ;
    case hipdnn_data_sdk::data_objects::PointwiseMode::CMP_GE:
        return hipdnn_frontend::PointwiseMode::CMP_GE;
    case hipdnn_data_sdk::data_objects::PointwiseMode::CMP_GT:
        return hipdnn_frontend::PointwiseMode::CMP_GT;
    case hipdnn_data_sdk::data_objects::PointwiseMode::CMP_LE:
        return hipdnn_frontend::PointwiseMode::CMP_LE;
    case hipdnn_data_sdk::data_objects::PointwiseMode::CMP_LT:
        return hipdnn_frontend::PointwiseMode::CMP_LT;
    case hipdnn_data_sdk::data_objects::PointwiseMode::CMP_NEQ:
        return hipdnn_frontend::PointwiseMode::CMP_NEQ;
    case hipdnn_data_sdk::data_objects::PointwiseMode::DIV:
        return hipdnn_frontend::PointwiseMode::DIV;
    case hipdnn_data_sdk::data_objects::PointwiseMode::ELU_BWD:
        return hipdnn_frontend::PointwiseMode::ELU_BWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::ELU_FWD:
        return hipdnn_frontend::PointwiseMode::ELU_FWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::ERF:
        return hipdnn_frontend::PointwiseMode::ERF;
    case hipdnn_data_sdk::data_objects::PointwiseMode::EXP:
        return hipdnn_frontend::PointwiseMode::EXP;
    case hipdnn_data_sdk::data_objects::PointwiseMode::FLOOR:
        return hipdnn_frontend::PointwiseMode::FLOOR;
    case hipdnn_data_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_BWD:
        return hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_BWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD:
        return hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_FWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::GELU_BWD:
        return hipdnn_frontend::PointwiseMode::GELU_BWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::GELU_FWD:
        return hipdnn_frontend::PointwiseMode::GELU_FWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::GEN_INDEX:
        return hipdnn_frontend::PointwiseMode::GEN_INDEX;
    case hipdnn_data_sdk::data_objects::PointwiseMode::IDENTITY:
        return hipdnn_frontend::PointwiseMode::IDENTITY;
    case hipdnn_data_sdk::data_objects::PointwiseMode::LOG:
        return hipdnn_frontend::PointwiseMode::LOG;
    case hipdnn_data_sdk::data_objects::PointwiseMode::LOGICAL_AND:
        return hipdnn_frontend::PointwiseMode::LOGICAL_AND;
    case hipdnn_data_sdk::data_objects::PointwiseMode::LOGICAL_NOT:
        return hipdnn_frontend::PointwiseMode::LOGICAL_NOT;
    case hipdnn_data_sdk::data_objects::PointwiseMode::LOGICAL_OR:
        return hipdnn_frontend::PointwiseMode::LOGICAL_OR;
    case hipdnn_data_sdk::data_objects::PointwiseMode::MAX_OP:
        return hipdnn_frontend::PointwiseMode::MAX;
    case hipdnn_data_sdk::data_objects::PointwiseMode::MIN_OP:
        return hipdnn_frontend::PointwiseMode::MIN;
    case hipdnn_data_sdk::data_objects::PointwiseMode::MUL:
        return hipdnn_frontend::PointwiseMode::MUL;
    case hipdnn_data_sdk::data_objects::PointwiseMode::NEG:
        return hipdnn_frontend::PointwiseMode::NEG;
    case hipdnn_data_sdk::data_objects::PointwiseMode::RECIPROCAL:
        return hipdnn_frontend::PointwiseMode::RECIPROCAL;
    case hipdnn_data_sdk::data_objects::PointwiseMode::RELU_BWD:
        return hipdnn_frontend::PointwiseMode::RELU_BWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD:
        return hipdnn_frontend::PointwiseMode::RELU_FWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::RSQRT:
        return hipdnn_frontend::PointwiseMode::RSQRT;
    case hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_BWD:
        return hipdnn_frontend::PointwiseMode::SIGMOID_BWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::SIGMOID_FWD:
        return hipdnn_frontend::PointwiseMode::SIGMOID_FWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::SIN:
        return hipdnn_frontend::PointwiseMode::SIN;
    case hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD:
        return hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD:
        return hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::SQRT:
        return hipdnn_frontend::PointwiseMode::SQRT;
    case hipdnn_data_sdk::data_objects::PointwiseMode::SUB:
        return hipdnn_frontend::PointwiseMode::SUB;
    case hipdnn_data_sdk::data_objects::PointwiseMode::SWISH_BWD:
        return hipdnn_frontend::PointwiseMode::SWISH_BWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::SWISH_FWD:
        return hipdnn_frontend::PointwiseMode::SWISH_FWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::TAN:
        return hipdnn_frontend::PointwiseMode::TAN;
    case hipdnn_data_sdk::data_objects::PointwiseMode::TANH_BWD:
        return hipdnn_frontend::PointwiseMode::TANH_BWD;
    case hipdnn_data_sdk::data_objects::PointwiseMode::TANH_FWD:
        return hipdnn_frontend::PointwiseMode::TANH_FWD;
    default:
        return hipdnn_frontend::PointwiseMode::NOT_SET;
    }
}

/// @brief Convert frontend HeuristicMode to backend heuristic mode
inline hipdnnBackendHeurMode_t toBackendType(const HeuristicMode& type)
{
    switch(type)
    {
    case HeuristicMode::FALLBACK:
    default:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    }
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
    default:
        return "unknown";
    }
}

/// @brief Stream insertion operator for DataType
inline std::ostream& operator<<(std::ostream& os, const DataType& type)
{
    return os << to_string(type);
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
    default:
        return "unknown";
    }
}

/// @brief Stream insertion operator for HeuristicMode
inline std::ostream& operator<<(std::ostream& os, const HeuristicMode& mode)
{
    return os << to_string(mode);
}

/// @brief Convert frontend KnobValueType to SDK KnobValue
inline hipdnn_data_sdk::data_objects::KnobValue toSdkType(const KnobValueType& type)
{
    switch(type)
    {
    case KnobValueType::INT64:
        return hipdnn_data_sdk::data_objects::KnobValue::IntValue;
    case KnobValueType::FLOAT64:
        return hipdnn_data_sdk::data_objects::KnobValue::FloatValue;
    case KnobValueType::STRING:
        return hipdnn_data_sdk::data_objects::KnobValue::StringValue;
    default:
        return hipdnn_data_sdk::data_objects::KnobValue::NONE;
    }
}

/// @brief Convert SDK KnobValue to frontend KnobValueType
inline hipdnn_frontend::KnobValueType
    fromSdkType(const hipdnn_data_sdk::data_objects::KnobValue& type)
{
    switch(type)
    {
    case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
        return hipdnn_frontend::KnobValueType::INT64;
    case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
        return hipdnn_frontend::KnobValueType::FLOAT64;
    case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
        return hipdnn_frontend::KnobValueType::STRING;
    default:
        return hipdnn_frontend::KnobValueType::NOT_SET;
    }
}

/// @brief Convert frontend NormFwdPhase to SDK NormFwdPhase
inline hipdnn_data_sdk::data_objects::NormFwdPhase toSdkType(const NormFwdPhase& type)
{
    switch(type)
    {
    case NormFwdPhase::INFERENCE:
        return hipdnn_data_sdk::data_objects::NormFwdPhase::INFERENCE;
    case NormFwdPhase::TRAINING:
        return hipdnn_data_sdk::data_objects::NormFwdPhase::TRAINING;
    default:
        return hipdnn_data_sdk::data_objects::NormFwdPhase::NOT_SET;
    }
}

/// @brief Convert SDK NormFwdPhase to frontend NormFwdPhase
inline hipdnn_frontend::NormFwdPhase
    fromSdkType(const hipdnn_data_sdk::data_objects::NormFwdPhase& type)
{
    switch(type)
    {
    case hipdnn_data_sdk::data_objects::NormFwdPhase::INFERENCE:
        return hipdnn_frontend::NormFwdPhase::INFERENCE;
    case hipdnn_data_sdk::data_objects::NormFwdPhase::TRAINING:
        return hipdnn_frontend::NormFwdPhase::TRAINING;
    default:
        return hipdnn_frontend::NormFwdPhase::NOT_SET;
    }
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

/**
 * @brief Check if a pointwise mode is a unary operation (1 input)
 * @param mode The pointwise mode to check
 * @return true if the mode is unary, false otherwise
 */
inline bool isUnaryPointwiseMode(PointwiseMode mode)
{
    return hipdnn_data_sdk::utilities::isUnaryPointwiseMode(toSdkType(mode));
}

/**
 * @brief Check if a pointwise mode is a binary operation (2 inputs)
 * @param mode The pointwise mode to check
 * @return true if the mode is binary, false otherwise
 */
inline bool isBinaryPointwiseMode(PointwiseMode mode)
{
    return hipdnn_data_sdk::utilities::isBinaryPointwiseMode(toSdkType(mode));
}

/**
 * @brief Check if a pointwise mode is a ternary operation (3 inputs)
 * @param mode The pointwise mode to check
 * @return true if the mode is ternary, false otherwise
 */
inline bool isTernaryPointwiseMode(PointwiseMode mode)
{
    return hipdnn_data_sdk::utilities::isTernaryPointwiseMode(toSdkType(mode));
}

/// @brief Get the bitset of all unary pointwise modes
inline const auto& getUnaryModesBitset()
{
    return hipdnn_data_sdk::utilities::getUnaryModesBitset();
}

/// @brief Get the bitset of all binary pointwise modes
inline const auto& getBinaryModesBitset()
{
    return hipdnn_data_sdk::utilities::getBinaryModesBitset();
}

/// @brief Get the bitset of all ternary pointwise modes
inline const auto& getTernaryModesBitset()
{
    return hipdnn_data_sdk::utilities::getTernaryModesBitset();
}

} // namespace hipdnn_frontend

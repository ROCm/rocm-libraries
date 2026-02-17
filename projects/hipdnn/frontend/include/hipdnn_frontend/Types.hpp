// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <HipdnnBackendHeuristicType.h>
#include <hipdnn_data_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PointwiseValidation.hpp>

#include <bitset>
#include <ostream>
#include <set>
#include <string>
#include <variant>

namespace hipdnn_frontend
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_data_sdk::types::fp8_e5m2;
using hipdnn_data_sdk::types::half;

enum class ConvolutionMode
{
    NOT_SET = 0,
    CROSS_CORRELATION = 1,
    CONVOLUTION = 2
};
typedef ConvolutionMode ConvolutionMode_t; // NOLINT(readability-identifier-naming)

enum class PointwiseMode
{
    NOT_SET = 0,
    ABS = 1,
    ADD = 2,
    ADD_SQUARE = 3,
    BINARY_SELECT = 4,
    CEIL = 5,
    CMP_EQ = 6,
    CMP_GE = 7,
    CMP_GT = 8,
    CMP_LE = 9,
    CMP_LT = 10,
    CMP_NEQ = 11,
    DIV = 12,
    ELU_BWD = 13,
    ELU_FWD = 14,
    ERF = 15,
    EXP = 16,
    FLOOR = 17,
    GELU_APPROX_TANH_BWD = 18,
    GELU_APPROX_TANH_FWD = 19,
    GELU_BWD = 20,
    GELU_FWD = 21,
    GEN_INDEX = 22,
    IDENTITY = 23,
    LOG = 24,
    LOGICAL_AND = 25,
    LOGICAL_NOT = 26,
    LOGICAL_OR = 27,
    MAX = 28,
    MIN = 29,
    MUL = 30,
    NEG = 31,
    RECIPROCAL = 32,
    RELU_BWD = 33,
    RELU_FWD = 34,
    RSQRT = 35,
    SIGMOID_BWD = 36,
    SIGMOID_FWD = 37,
    SIN = 38,
    SOFTPLUS_BWD = 39,
    SOFTPLUS_FWD = 40,
    SQRT = 41,
    SUB = 42,
    SWISH_BWD = 43,
    SWISH_FWD = 44,
    TAN = 45,
    TANH_BWD = 46,
    TANH_FWD = 47,
};
typedef PointwiseMode PointwiseMode_t; // NOLINT(readability-identifier-naming)

enum class DataType
{
    NOT_SET = 0,
    FLOAT = 1,
    HALF = 2,
    BFLOAT16 = 3,
    DOUBLE = 4,
    UINT8 = 5,
    INT32 = 6,
    INT8 = 7,
    FP8_E4M3 = 8,
    FP8_E5M2 = 9,
};
typedef DataType DataType_t; // NOLINT(readability-identifier-naming)

enum class HeuristicMode
{
    FALLBACK,
};
typedef HeuristicMode HeurMode_t; // NOLINT(readability-identifier-naming)

enum class BuildPlanPolicy
{
    HEURISTICS_CHOICE, // Use heuristics to select the best plan
    ALL // Build all available plans (currently unused)
};
typedef BuildPlanPolicy BuildPlanPolicy_t; // NOLINT(readability-identifier-naming)

enum class KnobValueType
{
    NOT_SET = 0,
    INT64 = 1,
    FLOAT64 = 2,
    STRING = 3,
};
typedef KnobValueType KnobValueType_t; // NOLINT(readability-identifier-naming)

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

inline hipdnnBackendHeurMode_t toBackendType(const HeuristicMode& type)
{
    switch(type)
    {
    case HeuristicMode::FALLBACK:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    default:
        return hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK;
    }
}

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

inline std::ostream& operator<<(std::ostream& os, const DataType& type)
{
    return os << to_string(type);
}

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

inline std::ostream& operator<<(std::ostream& os, const BuildPlanPolicy& policy)
{
    return os << to_string(policy);
}

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

inline std::ostream& operator<<(std::ostream& os, const HeuristicMode& mode)
{
    return os << to_string(mode);
}

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

inline std::ostream& operator<<(std::ostream& os, const KnobValueType& type)
{
    return os << to_string(type);
}

// Helper function to get KnobValueType from a variant
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

// Frontend functions delegate to SDK for single source of truth
// Convert frontend PointwiseMode to SDK type and call SDK validation functions

inline bool isUnaryPointwiseMode(PointwiseMode mode)
{
    return hipdnn_data_sdk::utilities::isUnaryPointwiseMode(toSdkType(mode));
}

inline bool isBinaryPointwiseMode(PointwiseMode mode)
{
    return hipdnn_data_sdk::utilities::isBinaryPointwiseMode(toSdkType(mode));
}

inline bool isTernaryPointwiseMode(PointwiseMode mode)
{
    return hipdnn_data_sdk::utilities::isTernaryPointwiseMode(toSdkType(mode));
}

// Expose SDK bitset functions for compatibility (delegate to SDK)
inline const auto& getUnaryModesBitset()
{
    return hipdnn_data_sdk::utilities::getUnaryModesBitset();
}

inline const auto& getBinaryModesBitset()
{
    return hipdnn_data_sdk::utilities::getBinaryModesBitset();
}

inline const auto& getTernaryModesBitset()
{
    return hipdnn_data_sdk::utilities::getTernaryModesBitset();
}

} // namespace hipdnn_frontend

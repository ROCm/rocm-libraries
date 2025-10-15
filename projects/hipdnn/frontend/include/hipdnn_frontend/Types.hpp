// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <HipdnnBackendHeuristicType.h>
#include <hipdnn_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

#include <bitset>
#include <set>

namespace hipdnn_frontend
{

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
};
typedef DataType DataType_t; // NOLINT(readability-identifier-naming)

enum class HeuristicMode
{
    FALLBACK,
};
typedef HeuristicMode HeurMode_t; // NOLINT(readability-identifier-naming)

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
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
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
    else
    {
        return DataType::NOT_SET;
    }
}

inline hipdnn_sdk::data_objects::ConvMode toSdkType(const ConvolutionMode& type)
{
    switch(type)
    {
    case ConvolutionMode::CROSS_CORRELATION:
        return hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    case ConvolutionMode::CONVOLUTION:
        return hipdnn_sdk::data_objects::ConvMode::CONVOLUTION;
    default:
        return hipdnn_sdk::data_objects::ConvMode::UNSET;
    }
}

inline hipdnn_sdk::data_objects::DataType toSdkType(const DataType& type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return hipdnn_sdk::data_objects::DataType::FLOAT;
    case DataType::HALF:
        return hipdnn_sdk::data_objects::DataType::HALF;
    case DataType::BFLOAT16:
        return hipdnn_sdk::data_objects::DataType::BFLOAT16;
    case DataType::DOUBLE:
        return hipdnn_sdk::data_objects::DataType::DOUBLE;
    case DataType::UINT8:
        return hipdnn_sdk::data_objects::DataType::UINT8;
    case DataType::INT32:
        return hipdnn_sdk::data_objects::DataType::INT32;
    default:
        return hipdnn_sdk::data_objects::DataType::UNSET;
    }
}

inline hipdnn_frontend::DataType fromSdkType(const hipdnn_sdk::data_objects::DataType& type)
{
    switch(type)
    {
    case hipdnn_sdk::data_objects::DataType::FLOAT:
        return hipdnn_frontend::DataType::FLOAT;
    case hipdnn_sdk::data_objects::DataType::HALF:
        return hipdnn_frontend::DataType::HALF;
    case hipdnn_sdk::data_objects::DataType::BFLOAT16:
        return hipdnn_frontend::DataType::BFLOAT16;
    case hipdnn_sdk::data_objects::DataType::DOUBLE:
        return hipdnn_frontend::DataType::DOUBLE;
    case hipdnn_sdk::data_objects::DataType::UINT8:
        return hipdnn_frontend::DataType::UINT8;
    case hipdnn_sdk::data_objects::DataType::INT32:
        return hipdnn_frontend::DataType::INT32;
    default:
        return hipdnn_frontend::DataType::NOT_SET;
    }
}

inline hipdnn_sdk::data_objects::PointwiseMode toSdkType(const PointwiseMode& type)
{
    switch(type)
    {
    case PointwiseMode::ABS:
        return hipdnn_sdk::data_objects::PointwiseMode::ABS;
    case PointwiseMode::ADD:
        return hipdnn_sdk::data_objects::PointwiseMode::ADD;
    case PointwiseMode::ADD_SQUARE:
        return hipdnn_sdk::data_objects::PointwiseMode::ADD_SQUARE;
    case PointwiseMode::BINARY_SELECT:
        return hipdnn_sdk::data_objects::PointwiseMode::BINARY_SELECT;
    case PointwiseMode::CEIL:
        return hipdnn_sdk::data_objects::PointwiseMode::CEIL;
    case PointwiseMode::CMP_EQ:
        return hipdnn_sdk::data_objects::PointwiseMode::CMP_EQ;
    case PointwiseMode::CMP_GE:
        return hipdnn_sdk::data_objects::PointwiseMode::CMP_GE;
    case PointwiseMode::CMP_GT:
        return hipdnn_sdk::data_objects::PointwiseMode::CMP_GT;
    case PointwiseMode::CMP_LE:
        return hipdnn_sdk::data_objects::PointwiseMode::CMP_LE;
    case PointwiseMode::CMP_LT:
        return hipdnn_sdk::data_objects::PointwiseMode::CMP_LT;
    case PointwiseMode::CMP_NEQ:
        return hipdnn_sdk::data_objects::PointwiseMode::CMP_NEQ;
    case PointwiseMode::DIV:
        return hipdnn_sdk::data_objects::PointwiseMode::DIV;
    case PointwiseMode::ELU_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::ELU_BWD;
    case PointwiseMode::ELU_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::ELU_FWD;
    case PointwiseMode::ERF:
        return hipdnn_sdk::data_objects::PointwiseMode::ERF;
    case PointwiseMode::EXP:
        return hipdnn_sdk::data_objects::PointwiseMode::EXP;
    case PointwiseMode::FLOOR:
        return hipdnn_sdk::data_objects::PointwiseMode::FLOOR;
    case PointwiseMode::GELU_APPROX_TANH_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_BWD;
    case PointwiseMode::GELU_APPROX_TANH_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD;
    case PointwiseMode::GELU_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::GELU_BWD;
    case PointwiseMode::GELU_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::GELU_FWD;
    case PointwiseMode::GEN_INDEX:
        return hipdnn_sdk::data_objects::PointwiseMode::GEN_INDEX;
    case PointwiseMode::IDENTITY:
        return hipdnn_sdk::data_objects::PointwiseMode::IDENTITY;
    case PointwiseMode::LOG:
        return hipdnn_sdk::data_objects::PointwiseMode::LOG;
    case PointwiseMode::LOGICAL_AND:
        return hipdnn_sdk::data_objects::PointwiseMode::LOGICAL_AND;
    case PointwiseMode::LOGICAL_NOT:
        return hipdnn_sdk::data_objects::PointwiseMode::LOGICAL_NOT;
    case PointwiseMode::LOGICAL_OR:
        return hipdnn_sdk::data_objects::PointwiseMode::LOGICAL_OR;
    case PointwiseMode::MAX:
        return hipdnn_sdk::data_objects::PointwiseMode::MAX_OP;
    case PointwiseMode::MIN:
        return hipdnn_sdk::data_objects::PointwiseMode::MIN_OP;
    case PointwiseMode::MUL:
        return hipdnn_sdk::data_objects::PointwiseMode::MUL;
    case PointwiseMode::NEG:
        return hipdnn_sdk::data_objects::PointwiseMode::NEG;
    case PointwiseMode::RECIPROCAL:
        return hipdnn_sdk::data_objects::PointwiseMode::RECIPROCAL;
    case PointwiseMode::RELU_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD;
    case PointwiseMode::RELU_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD;
    case PointwiseMode::RSQRT:
        return hipdnn_sdk::data_objects::PointwiseMode::RSQRT;
    case PointwiseMode::SIGMOID_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_BWD;
    case PointwiseMode::SIGMOID_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD;
    case PointwiseMode::SIN:
        return hipdnn_sdk::data_objects::PointwiseMode::SIN;
    case PointwiseMode::SOFTPLUS_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD;
    case PointwiseMode::SOFTPLUS_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD;
    case PointwiseMode::SQRT:
        return hipdnn_sdk::data_objects::PointwiseMode::SQRT;
    case PointwiseMode::SUB:
        return hipdnn_sdk::data_objects::PointwiseMode::SUB;
    case PointwiseMode::SWISH_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::SWISH_BWD;
    case PointwiseMode::SWISH_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::SWISH_FWD;
    case PointwiseMode::TAN:
        return hipdnn_sdk::data_objects::PointwiseMode::TAN;
    case PointwiseMode::TANH_BWD:
        return hipdnn_sdk::data_objects::PointwiseMode::TANH_BWD;
    case PointwiseMode::TANH_FWD:
        return hipdnn_sdk::data_objects::PointwiseMode::TANH_FWD;
    default:
        return hipdnn_sdk::data_objects::PointwiseMode::UNSET;
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
    default:
        return "unknown";
    }
}

inline std::ostream& operator<<(std::ostream& os, const DataType& type)
{
    return os << to_string(type);
}

// Frontend functions delegate to SDK for single source of truth
// Convert frontend PointwiseMode to SDK type and call SDK validation functions

inline bool isUnaryPointwiseMode(PointwiseMode mode)
{
    return hipdnn_sdk::utilities::isUnaryPointwiseMode(toSdkType(mode));
}

inline bool isBinaryPointwiseMode(PointwiseMode mode)
{
    return hipdnn_sdk::utilities::isBinaryPointwiseMode(toSdkType(mode));
}

inline bool isTernaryPointwiseMode(PointwiseMode mode)
{
    return hipdnn_sdk::utilities::isTernaryPointwiseMode(toSdkType(mode));
}

// Expose SDK bitset functions for compatibility (delegate to SDK)
inline const auto& getUnaryModesBitset()
{
    return hipdnn_sdk::utilities::getUnaryModesBitset();
}

inline const auto& getBinaryModesBitset()
{
    return hipdnn_sdk::utilities::getBinaryModesBitset();
}

inline const auto& getTernaryModesBitset()
{
    return hipdnn_sdk::utilities::getTernaryModesBitset();
}

} // namespace hipdnn_frontend

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

namespace hipdnn_test_sdk::utilities
{

// Conversions between frontend and SDK enum types for use in test utilities.
// If new data types are added, update the mappings below.

/// Convert frontend DataType to SDK DataType for use in test utilities
inline hipdnn_flatbuffers_sdk::data_objects::DataType
    frontendToSdkDataType(const hipdnn_frontend::DataType& type)
{
    switch(type)
    {
    case hipdnn_frontend::DataType::FLOAT:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    case hipdnn_frontend::DataType::HALF:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    case hipdnn_frontend::DataType::BFLOAT16:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    case hipdnn_frontend::DataType::DOUBLE:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE;
    case hipdnn_frontend::DataType::UINT8:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8;
    case hipdnn_frontend::DataType::INT32:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    case hipdnn_frontend::DataType::INT8:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::INT8;
    case hipdnn_frontend::DataType::FP8_E4M3:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3;
    case hipdnn_frontend::DataType::FP8_E5M2:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2;
    case hipdnn_frontend::DataType::FP8_E8M0:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0;
    case hipdnn_frontend::DataType::FP4_E2M1:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1;
    case hipdnn_frontend::DataType::INT4:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::INT4;
    case hipdnn_frontend::DataType::FP6_E2M3:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3;
    case hipdnn_frontend::DataType::FP6_E3M2:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2;
    case hipdnn_frontend::DataType::INT64:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::INT64;
    default:
        return hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    }
}

/// Convert SDK DataType to frontend DataType for use in test utilities
inline hipdnn_frontend::DataType
    sdkToFrontendDataType(const hipdnn_flatbuffers_sdk::data_objects::DataType& type)
{
    switch(type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return hipdnn_frontend::DataType::FLOAT;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
        return hipdnn_frontend::DataType::HALF;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return hipdnn_frontend::DataType::BFLOAT16;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE:
        return hipdnn_frontend::DataType::DOUBLE;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8:
        return hipdnn_frontend::DataType::UINT8;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT32:
        return hipdnn_frontend::DataType::INT32;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT8:
        return hipdnn_frontend::DataType::INT8;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3:
        return hipdnn_frontend::DataType::FP8_E4M3;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2:
        return hipdnn_frontend::DataType::FP8_E5M2;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0:
        return hipdnn_frontend::DataType::FP8_E8M0;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1:
        return hipdnn_frontend::DataType::FP4_E2M1;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT4:
        return hipdnn_frontend::DataType::INT4;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3:
        return hipdnn_frontend::DataType::FP6_E2M3;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2:
        return hipdnn_frontend::DataType::FP6_E3M2;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT64:
        return hipdnn_frontend::DataType::INT64;
    default:
        return hipdnn_frontend::DataType::NOT_SET;
    }
}

/// Convert frontend PointwiseMode to SDK PointwiseMode for use in test utilities
inline hipdnn_flatbuffers_sdk::data_objects::PointwiseMode
    frontendToSdkPointwiseMode(const hipdnn_frontend::PointwiseMode& mode)
{
    switch(mode)
    {
    case hipdnn_frontend::PointwiseMode::NOT_SET:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET;
    case hipdnn_frontend::PointwiseMode::ABS:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ABS;
    case hipdnn_frontend::PointwiseMode::ADD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD;
    case hipdnn_frontend::PointwiseMode::ADD_SQUARE:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD_SQUARE;
    case hipdnn_frontend::PointwiseMode::BINARY_SELECT:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::BINARY_SELECT;
    case hipdnn_frontend::PointwiseMode::CEIL:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CEIL;
    case hipdnn_frontend::PointwiseMode::CMP_EQ:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_EQ;
    case hipdnn_frontend::PointwiseMode::CMP_GE:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_GE;
    case hipdnn_frontend::PointwiseMode::CMP_GT:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_GT;
    case hipdnn_frontend::PointwiseMode::CMP_LE:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_LE;
    case hipdnn_frontend::PointwiseMode::CMP_LT:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_LT;
    case hipdnn_frontend::PointwiseMode::CMP_NEQ:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_NEQ;
    case hipdnn_frontend::PointwiseMode::DIV:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::DIV;
    case hipdnn_frontend::PointwiseMode::ELU_BWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ELU_BWD;
    case hipdnn_frontend::PointwiseMode::ELU_FWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ELU_FWD;
    case hipdnn_frontend::PointwiseMode::ERF:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ERF;
    case hipdnn_frontend::PointwiseMode::EXP:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::EXP;
    case hipdnn_frontend::PointwiseMode::FLOOR:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::FLOOR;
    case hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_BWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_BWD;
    case hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_FWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD;
    case hipdnn_frontend::PointwiseMode::GELU_BWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_BWD;
    case hipdnn_frontend::PointwiseMode::GELU_FWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_FWD;
    case hipdnn_frontend::PointwiseMode::GEN_INDEX:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GEN_INDEX;
    case hipdnn_frontend::PointwiseMode::IDENTITY:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY;
    case hipdnn_frontend::PointwiseMode::LOG:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOG;
    case hipdnn_frontend::PointwiseMode::LOGICAL_AND:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOGICAL_AND;
    case hipdnn_frontend::PointwiseMode::LOGICAL_NOT:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOGICAL_NOT;
    case hipdnn_frontend::PointwiseMode::LOGICAL_OR:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOGICAL_OR;
    case hipdnn_frontend::PointwiseMode::MAX:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MAX_OP;
    case hipdnn_frontend::PointwiseMode::MIN:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MIN_OP;
    case hipdnn_frontend::PointwiseMode::MUL:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL;
    case hipdnn_frontend::PointwiseMode::NEG:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::NEG;
    case hipdnn_frontend::PointwiseMode::RECIPROCAL:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RECIPROCAL;
    case hipdnn_frontend::PointwiseMode::RELU_BWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD;
    case hipdnn_frontend::PointwiseMode::RELU_FWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD;
    case hipdnn_frontend::PointwiseMode::RSQRT:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RSQRT;
    case hipdnn_frontend::PointwiseMode::SIGMOID_BWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_BWD;
    case hipdnn_frontend::PointwiseMode::SIGMOID_FWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD;
    case hipdnn_frontend::PointwiseMode::SIN:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIN;
    case hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD;
    case hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD;
    case hipdnn_frontend::PointwiseMode::SQRT:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SQRT;
    case hipdnn_frontend::PointwiseMode::SUB:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB;
    case hipdnn_frontend::PointwiseMode::SWISH_BWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_BWD;
    case hipdnn_frontend::PointwiseMode::SWISH_FWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD;
    case hipdnn_frontend::PointwiseMode::TAN:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TAN;
    case hipdnn_frontend::PointwiseMode::TANH_BWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_BWD;
    case hipdnn_frontend::PointwiseMode::TANH_FWD:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_FWD;
    default:
        return hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET;
    }
}

/// Convert SDK PointwiseMode to frontend PointwiseMode for use in test utilities
inline hipdnn_frontend::PointwiseMode
    sdkToFrontendPointwiseMode(const hipdnn_flatbuffers_sdk::data_objects::PointwiseMode& mode)
{
    switch(mode)
    {
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET:
        return hipdnn_frontend::PointwiseMode::NOT_SET;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ABS:
        return hipdnn_frontend::PointwiseMode::ABS;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD:
        return hipdnn_frontend::PointwiseMode::ADD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD_SQUARE:
        return hipdnn_frontend::PointwiseMode::ADD_SQUARE;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::BINARY_SELECT:
        return hipdnn_frontend::PointwiseMode::BINARY_SELECT;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CEIL:
        return hipdnn_frontend::PointwiseMode::CEIL;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_EQ:
        return hipdnn_frontend::PointwiseMode::CMP_EQ;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_GE:
        return hipdnn_frontend::PointwiseMode::CMP_GE;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_GT:
        return hipdnn_frontend::PointwiseMode::CMP_GT;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_LE:
        return hipdnn_frontend::PointwiseMode::CMP_LE;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_LT:
        return hipdnn_frontend::PointwiseMode::CMP_LT;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CMP_NEQ:
        return hipdnn_frontend::PointwiseMode::CMP_NEQ;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::DIV:
        return hipdnn_frontend::PointwiseMode::DIV;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ELU_BWD:
        return hipdnn_frontend::PointwiseMode::ELU_BWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ELU_FWD:
        return hipdnn_frontend::PointwiseMode::ELU_FWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ERF:
        return hipdnn_frontend::PointwiseMode::ERF;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::EXP:
        return hipdnn_frontend::PointwiseMode::EXP;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::FLOOR:
        return hipdnn_frontend::PointwiseMode::FLOOR;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_BWD:
        return hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_BWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD:
        return hipdnn_frontend::PointwiseMode::GELU_APPROX_TANH_FWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_BWD:
        return hipdnn_frontend::PointwiseMode::GELU_BWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_FWD:
        return hipdnn_frontend::PointwiseMode::GELU_FWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GEN_INDEX:
        return hipdnn_frontend::PointwiseMode::GEN_INDEX;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY:
        return hipdnn_frontend::PointwiseMode::IDENTITY;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOG:
        return hipdnn_frontend::PointwiseMode::LOG;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOGICAL_AND:
        return hipdnn_frontend::PointwiseMode::LOGICAL_AND;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOGICAL_NOT:
        return hipdnn_frontend::PointwiseMode::LOGICAL_NOT;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::LOGICAL_OR:
        return hipdnn_frontend::PointwiseMode::LOGICAL_OR;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MAX_OP:
        return hipdnn_frontend::PointwiseMode::MAX;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MIN_OP:
        return hipdnn_frontend::PointwiseMode::MIN;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL:
        return hipdnn_frontend::PointwiseMode::MUL;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::NEG:
        return hipdnn_frontend::PointwiseMode::NEG;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RECIPROCAL:
        return hipdnn_frontend::PointwiseMode::RECIPROCAL;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD:
        return hipdnn_frontend::PointwiseMode::RELU_BWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD:
        return hipdnn_frontend::PointwiseMode::RELU_FWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RSQRT:
        return hipdnn_frontend::PointwiseMode::RSQRT;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_BWD:
        return hipdnn_frontend::PointwiseMode::SIGMOID_BWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD:
        return hipdnn_frontend::PointwiseMode::SIGMOID_FWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIN:
        return hipdnn_frontend::PointwiseMode::SIN;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD:
        return hipdnn_frontend::PointwiseMode::SOFTPLUS_BWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD:
        return hipdnn_frontend::PointwiseMode::SOFTPLUS_FWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SQRT:
        return hipdnn_frontend::PointwiseMode::SQRT;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB:
        return hipdnn_frontend::PointwiseMode::SUB;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_BWD:
        return hipdnn_frontend::PointwiseMode::SWISH_BWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD:
        return hipdnn_frontend::PointwiseMode::SWISH_FWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TAN:
        return hipdnn_frontend::PointwiseMode::TAN;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_BWD:
        return hipdnn_frontend::PointwiseMode::TANH_BWD;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_FWD:
        return hipdnn_frontend::PointwiseMode::TANH_FWD;
    default:
        return hipdnn_frontend::PointwiseMode::NOT_SET;
    }
}

/// Create a Data SDK ITensor from a frontend DataType enum
inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    createTensor(hipdnn_frontend::DataType dataType,
                 const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& strides)
{
    using namespace hipdnn_data_sdk::utilities;
    using namespace hipdnn_data_sdk::types;
    switch(dataType)
    {
    case hipdnn_frontend::DataType::FLOAT:
        return std::make_unique<Tensor<float>>(dims, strides);
    case hipdnn_frontend::DataType::HALF:
        return std::make_unique<Tensor<half>>(dims, strides);
    case hipdnn_frontend::DataType::BFLOAT16:
        return std::make_unique<Tensor<bfloat16>>(dims, strides);
    case hipdnn_frontend::DataType::DOUBLE:
        return std::make_unique<Tensor<double>>(dims, strides);
    case hipdnn_frontend::DataType::UINT8:
        return std::make_unique<Tensor<uint8_t>>(dims, strides);
    case hipdnn_frontend::DataType::INT32:
        return std::make_unique<Tensor<int32_t>>(dims, strides);
    case hipdnn_frontend::DataType::INT8:
        return std::make_unique<Tensor<int8_t>>(dims, strides);
    case hipdnn_frontend::DataType::FP8_E4M3:
        return std::make_unique<Tensor<fp8_e4m3>>(dims, strides);
    case hipdnn_frontend::DataType::FP8_E5M2:
        return std::make_unique<Tensor<fp8_e5m2>>(dims, strides);
    case hipdnn_frontend::DataType::INT64:
        return std::make_unique<Tensor<int64_t>>(dims, strides);
    case hipdnn_frontend::DataType::FP8_E8M0:
        return std::make_unique<Tensor<fp8_e8m0>>(dims, strides);
    case hipdnn_frontend::DataType::FP4_E2M1:
        return std::make_unique<Tensor<fp4_e2m1>>(dims, strides);
    case hipdnn_frontend::DataType::INT4:
        return std::make_unique<Tensor<uint8_t>>(dims, strides);
    case hipdnn_frontend::DataType::FP6_E2M3:
        return std::make_unique<Tensor<fp6_e2m3>>(dims, strides);
    case hipdnn_frontend::DataType::FP6_E3M2:
        return std::make_unique<Tensor<fp6_e3m2>>(dims, strides);
    default:
        throw std::runtime_error("Unsupported data type for tensor");
    }
}

/// Create a Data SDK ITensor from frontend TensorAttributes
inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    createTensorFromAttribute(const hipdnn_frontend::graph::TensorAttributes& attribute)
{
    return createTensor(attribute.get_data_type(), attribute.get_dim(), attribute.get_stride());
}

} // namespace hipdnn_test_sdk::utilities

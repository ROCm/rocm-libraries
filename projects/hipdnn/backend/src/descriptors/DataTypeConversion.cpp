// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DataTypeConversion.hpp"

namespace hipdnn_backend
{

hipdnn_flatbuffers_sdk::data_objects::DataType toSdkDataType(hipdnnDataType_t type)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    switch(type)
    {
    case HIPDNN_DATA_FLOAT:
        return DataType::FLOAT;
    case HIPDNN_DATA_DOUBLE:
        return DataType::DOUBLE;
    case HIPDNN_DATA_HALF:
        return DataType::HALF;
    case HIPDNN_DATA_INT8:
        return DataType::INT8;
    case HIPDNN_DATA_INT32:
        return DataType::INT32;
    case HIPDNN_DATA_UINT8:
        return DataType::UINT8;
    case HIPDNN_DATA_BFLOAT16:
        return DataType::BFLOAT16;
    case HIPDNN_DATA_FP8_E4M3:
        return DataType::FP8_E4M3;
    case HIPDNN_DATA_FP8_E5M2:
        return DataType::FP8_E5M2;
    case HIPDNN_DATA_FP8_E8M0:
        return DataType::FP8_E8M0;
    case HIPDNN_DATA_FP4_E2M1:
        return DataType::FP4_E2M1;
    case HIPDNN_DATA_INT4:
        return DataType::INT4;
    case HIPDNN_DATA_FP6_E2M3_EXT:
        return DataType::FP6_E2M3;
    case HIPDNN_DATA_FP6_E3M2_EXT:
        return DataType::FP6_E3M2;
    case HIPDNN_DATA_INT64:
        return DataType::INT64;
    case HIPDNN_DATA_BOOLEAN:
        return DataType::BOOLEAN;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported hipdnnDataType_t value");
    }
}

hipdnnDataType_t fromSdkDataType(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

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
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK DataType");
    }
}

int64_t getDataTypeByteSize(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(type)
    {
    case DataType::FLOAT:
        return 4;
    case DataType::DOUBLE:
        return 8;
    case DataType::HALF:
    case DataType::BFLOAT16:
        return 2;
    case DataType::INT32:
        return 4;
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::BOOLEAN:
        return 1;
    case DataType::INT64:
        return 8;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported DataType for byte size");
    }
}

hipdnn_flatbuffers_sdk::data_objects::ConvMode toSdkConvMode(hipdnnConvolutionMode_t mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;

    switch(mode)
    {
    case HIPDNN_CONVOLUTION:
        return ConvMode::CONVOLUTION;
    case HIPDNN_CROSS_CORRELATION:
        return ConvMode::CROSS_CORRELATION;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported hipdnnConvolutionMode_t value");
    }
}

hipdnnConvolutionMode_t fromSdkConvMode(hipdnn_flatbuffers_sdk::data_objects::ConvMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::ConvMode;

    switch(mode)
    {
    case ConvMode::CONVOLUTION:
        return HIPDNN_CONVOLUTION;
    case ConvMode::CROSS_CORRELATION:
        return HIPDNN_CROSS_CORRELATION;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK ConvMode value");
    }
}

hipdnn_flatbuffers_sdk::data_objects::PointwiseMode toSdkPointwiseMode(hipdnnPointwiseMode_t mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    switch(mode)
    {
    case HIPDNN_POINTWISE_ABS:
        return PointwiseMode::ABS;
    case HIPDNN_POINTWISE_ADD:
        return PointwiseMode::ADD;
    case HIPDNN_POINTWISE_ADD_SQUARE:
        return PointwiseMode::ADD_SQUARE;
    case HIPDNN_POINTWISE_BINARY_SELECT:
        return PointwiseMode::BINARY_SELECT;
    case HIPDNN_POINTWISE_CEIL:
        return PointwiseMode::CEIL;
    case HIPDNN_POINTWISE_CMP_EQ:
        return PointwiseMode::CMP_EQ;
    case HIPDNN_POINTWISE_CMP_GE:
        return PointwiseMode::CMP_GE;
    case HIPDNN_POINTWISE_CMP_GT:
        return PointwiseMode::CMP_GT;
    case HIPDNN_POINTWISE_CMP_LE:
        return PointwiseMode::CMP_LE;
    case HIPDNN_POINTWISE_CMP_LT:
        return PointwiseMode::CMP_LT;
    case HIPDNN_POINTWISE_CMP_NEQ:
        return PointwiseMode::CMP_NEQ;
    case HIPDNN_POINTWISE_DIV:
        return PointwiseMode::DIV;
    case HIPDNN_POINTWISE_ELU_BWD:
        return PointwiseMode::ELU_BWD;
    case HIPDNN_POINTWISE_ELU_FWD:
        return PointwiseMode::ELU_FWD;
    case HIPDNN_POINTWISE_ERF:
        return PointwiseMode::ERF;
    case HIPDNN_POINTWISE_EXP:
        return PointwiseMode::EXP;
    case HIPDNN_POINTWISE_FLOOR:
        return PointwiseMode::FLOOR;
    case HIPDNN_POINTWISE_GELU_APPROX_TANH_BWD:
        return PointwiseMode::GELU_APPROX_TANH_BWD;
    case HIPDNN_POINTWISE_GELU_APPROX_TANH_FWD:
        return PointwiseMode::GELU_APPROX_TANH_FWD;
    case HIPDNN_POINTWISE_GELU_BWD:
        return PointwiseMode::GELU_BWD;
    case HIPDNN_POINTWISE_GELU_FWD:
        return PointwiseMode::GELU_FWD;
    case HIPDNN_POINTWISE_GEN_INDEX:
        return PointwiseMode::GEN_INDEX;
    case HIPDNN_POINTWISE_IDENTITY:
        return PointwiseMode::IDENTITY;
    case HIPDNN_POINTWISE_LOG:
        return PointwiseMode::LOG;
    case HIPDNN_POINTWISE_LOGICAL_AND:
        return PointwiseMode::LOGICAL_AND;
    case HIPDNN_POINTWISE_LOGICAL_NOT:
        return PointwiseMode::LOGICAL_NOT;
    case HIPDNN_POINTWISE_LOGICAL_OR:
        return PointwiseMode::LOGICAL_OR;
    case HIPDNN_POINTWISE_MAX:
        return PointwiseMode::MAX_OP;
    case HIPDNN_POINTWISE_MIN:
        return PointwiseMode::MIN_OP;
    case HIPDNN_POINTWISE_MUL:
        return PointwiseMode::MUL;
    case HIPDNN_POINTWISE_NEG:
        return PointwiseMode::NEG;
    case HIPDNN_POINTWISE_RECIPROCAL:
        return PointwiseMode::RECIPROCAL;
    case HIPDNN_POINTWISE_RELU_BWD:
        return PointwiseMode::RELU_BWD;
    case HIPDNN_POINTWISE_RELU_FWD:
        return PointwiseMode::RELU_FWD;
    case HIPDNN_POINTWISE_RSQRT:
        return PointwiseMode::RSQRT;
    case HIPDNN_POINTWISE_SIGMOID_BWD:
        return PointwiseMode::SIGMOID_BWD;
    case HIPDNN_POINTWISE_SIGMOID_FWD:
        return PointwiseMode::SIGMOID_FWD;
    case HIPDNN_POINTWISE_SIN:
        return PointwiseMode::SIN;
    case HIPDNN_POINTWISE_SOFTPLUS_BWD:
        return PointwiseMode::SOFTPLUS_BWD;
    case HIPDNN_POINTWISE_SOFTPLUS_FWD:
        return PointwiseMode::SOFTPLUS_FWD;
    case HIPDNN_POINTWISE_SQRT:
        return PointwiseMode::SQRT;
    case HIPDNN_POINTWISE_SUB:
        return PointwiseMode::SUB;
    case HIPDNN_POINTWISE_SWISH_BWD:
        return PointwiseMode::SWISH_BWD;
    case HIPDNN_POINTWISE_SWISH_FWD:
        return PointwiseMode::SWISH_FWD;
    case HIPDNN_POINTWISE_TAN:
        return PointwiseMode::TAN;
    case HIPDNN_POINTWISE_TANH_BWD:
        return PointwiseMode::TANH_BWD;
    case HIPDNN_POINTWISE_TANH_FWD:
        return PointwiseMode::TANH_FWD;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported hipdnnPointwiseMode_t value");
    }
}

hipdnnPointwiseMode_t fromSdkPointwiseMode(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    switch(mode)
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
    case PointwiseMode::MAX_OP:
        return HIPDNN_POINTWISE_MAX;
    case PointwiseMode::MIN_OP:
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
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK PointwiseMode value");
    }
}

hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment
    toSdkDiagonalAlignment(hipdnnDiagonalAlignment_t mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;

    switch(mode)
    {
    case HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT:
        return DiagonalAlignment::TOP_LEFT;
    case HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT:
        return DiagonalAlignment::BOTTOM_RIGHT;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "Unsupported hipdnnDiagonalAlignment_t value");
    }
}

hipdnnDiagonalAlignment_t
    fromSdkDiagonalAlignment(hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;

    switch(mode)
    {
    case DiagonalAlignment::TOP_LEFT:
        return HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;
    case DiagonalAlignment::BOTTOM_RIGHT:
        return HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK DiagonalAlignment value");
    }
}

hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation
    toSdkAttentionImplementation(hipdnnAttentionImplementation_t mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;

    switch(mode)
    {
    case HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT:
        return AttentionImplementation::AUTO;
    case HIPDNN_ATTENTION_IMPLEMENTATION_COMPOSITE_EXT:
        return AttentionImplementation::COMPOSITE;
    case HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT:
        return AttentionImplementation::UNIFIED;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "Unsupported hipdnnAttentionImplementation_t value");
    }
}

hipdnnAttentionImplementation_t fromSdkAttentionImplementation(
    hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation;

    switch(mode)
    {
    case AttentionImplementation::AUTO:
        return HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT;
    case AttentionImplementation::COMPOSITE:
        return HIPDNN_ATTENTION_IMPLEMENTATION_COMPOSITE_EXT;
    case AttentionImplementation::UNIFIED:
        return HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "Unsupported SDK AttentionImplementation value");
    }
}

hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase toSdkNormFwdPhase(hipdnnNormFwdPhase_t phase)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;

    switch(phase)
    {
    case HIPDNN_NORM_FWD_INFERENCE:
        return NormFwdPhase::INFERENCE;
    case HIPDNN_NORM_FWD_TRAINING:
        return NormFwdPhase::TRAINING;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported hipdnnNormFwdPhase_t value");
    }
}

hipdnnNormFwdPhase_t fromSdkNormFwdPhase(hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase phase)
{
    using hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;

    switch(phase)
    {
    case NormFwdPhase::INFERENCE:
        return HIPDNN_NORM_FWD_INFERENCE;
    case NormFwdPhase::TRAINING:
        return HIPDNN_NORM_FWD_TRAINING;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK NormFwdPhase value");
    }
}

hipdnn_flatbuffers_sdk::data_objects::ReductionMode toSdkReductionMode(hipdnnReduceTensorOp_t mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;

    switch(mode)
    {
    case HIPDNN_REDUCE_TENSOR_ADD:
        return ReductionMode::ADD;
    case HIPDNN_REDUCE_TENSOR_MUL:
        return ReductionMode::MUL;
    case HIPDNN_REDUCE_TENSOR_MIN:
        return ReductionMode::MIN_OP;
    case HIPDNN_REDUCE_TENSOR_MAX:
        return ReductionMode::MAX_OP;
    case HIPDNN_REDUCE_TENSOR_AMAX:
        return ReductionMode::AMAX;
    case HIPDNN_REDUCE_TENSOR_AVG:
        return ReductionMode::AVG;
    case HIPDNN_REDUCE_TENSOR_NORM1:
        return ReductionMode::NORM1;
    case HIPDNN_REDUCE_TENSOR_NORM2:
        return ReductionMode::NORM2;
    case HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
        return ReductionMode::MUL_NO_ZEROS;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported hipdnnReduceTensorOp_t value");
    }
}

hipdnnReduceTensorOp_t
    fromSdkReductionMode(hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;

    switch(mode)
    {
    case ReductionMode::ADD:
        return HIPDNN_REDUCE_TENSOR_ADD;
    case ReductionMode::MUL:
        return HIPDNN_REDUCE_TENSOR_MUL;
    case ReductionMode::MIN_OP:
        return HIPDNN_REDUCE_TENSOR_MIN;
    case ReductionMode::MAX_OP:
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
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK ReductionMode value");
    }
}

hipdnn_flatbuffers_sdk::data_objects::ResampleMode toSdkResampleMode(hipdnnResampleMode_t mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::ResampleMode;

    switch(mode)
    {
    case HIPDNN_RESAMPLE_MAXPOOL:
        return ResampleMode::MAXPOOL;
    case HIPDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING:
        return ResampleMode::AVGPOOL_EXCLUDE_PADDING;
    case HIPDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING:
        return ResampleMode::AVGPOOL_INCLUDE_PADDING;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported hipdnnResampleMode_t value");
    }
}

hipdnnResampleMode_t fromSdkResampleMode(hipdnn_flatbuffers_sdk::data_objects::ResampleMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::ResampleMode;

    switch(mode)
    {
    case ResampleMode::MAXPOOL:
        return HIPDNN_RESAMPLE_MAXPOOL;
    case ResampleMode::AVGPOOL_EXCLUDE_PADDING:
        return HIPDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;
    case ResampleMode::AVGPOOL_INCLUDE_PADDING:
        return HIPDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK ResampleMode value");
    }
}

hipdnn_flatbuffers_sdk::data_objects::PaddingMode toSdkPaddingMode(hipdnnPaddingMode_t mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PaddingMode;

    switch(mode)
    {
    case HIPDNN_PADDING_NEG_INF_PAD:
        return PaddingMode::NEG_INF_PAD;
    case HIPDNN_PADDING_ZERO_PAD:
        return PaddingMode::ZERO_PAD;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported hipdnnPaddingMode_t value");
    }
}

hipdnnPaddingMode_t fromSdkPaddingMode(hipdnn_flatbuffers_sdk::data_objects::PaddingMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PaddingMode;

    switch(mode)
    {
    case PaddingMode::NEG_INF_PAD:
        return HIPDNN_PADDING_NEG_INF_PAD;
    case PaddingMode::ZERO_PAD:
        return HIPDNN_PADDING_ZERO_PAD;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK PaddingMode value");
    }
}

} // namespace hipdnn_backend

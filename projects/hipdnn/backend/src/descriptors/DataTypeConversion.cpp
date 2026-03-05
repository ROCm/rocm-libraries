// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DataTypeConversion.hpp"

namespace hipdnn_backend
{

hipdnn_data_sdk::data_objects::DataType toSdkDataType(hipdnnDataType_t type)
{
    using hipdnn_data_sdk::data_objects::DataType;

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
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported hipdnnDataType_t value");
    }
}

hipdnnDataType_t fromSdkDataType(hipdnn_data_sdk::data_objects::DataType type)
{
    using hipdnn_data_sdk::data_objects::DataType;

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
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK DataType");
    }
}

int64_t getDataTypeByteSize(hipdnn_data_sdk::data_objects::DataType type)
{
    using hipdnn_data_sdk::data_objects::DataType;
    switch(type)
    {
    case DataType::FLOAT:
        return 4;
    case DataType::DOUBLE:
        return 8;
    case DataType::HALF:
        return 2;
    case DataType::BFLOAT16:
        return 2;
    case DataType::INT32:
        return 4;
    case DataType::UINT8:
        return 1;
    case DataType::INT8:
        return 1;
    case DataType::FP8_E4M3:
        return 1;
    case DataType::FP8_E5M2:
        return 1;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported DataType for byte size");
    }
}

hipdnn_data_sdk::data_objects::ConvMode toSdkConvMode(hipdnnConvolutionMode_t mode)
{
    using hipdnn_data_sdk::data_objects::ConvMode;

    switch(mode)
    {
    case HIPDNN_CONVOLUTION_MODE_CONVOLUTION:
        return ConvMode::CONVOLUTION;
    case HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION:
        return ConvMode::CROSS_CORRELATION;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported hipdnnConvolutionMode_t value");
    }
}

hipdnnConvolutionMode_t fromSdkConvMode(hipdnn_data_sdk::data_objects::ConvMode mode)
{
    using hipdnn_data_sdk::data_objects::ConvMode;

    switch(mode)
    {
    case ConvMode::CONVOLUTION:
        return HIPDNN_CONVOLUTION_MODE_CONVOLUTION;
    case ConvMode::CROSS_CORRELATION:
        return HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION;
    default:
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, "Unsupported SDK ConvMode value");
    }
}

} // namespace hipdnn_backend

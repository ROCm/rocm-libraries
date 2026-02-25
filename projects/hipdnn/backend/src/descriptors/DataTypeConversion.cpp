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
        return DataType::UNSET;
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
        return -1;
    }
}

void setDataType(hipdnn_data_sdk::data_objects::DataType& target,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElements,
                 const char* errorPrefix)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_DATA_TYPE,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": attributeType mismatch");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": elementCount is not 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  std::string(errorPrefix) + ": arrayOfElements is null");
    target = toSdkDataType(*static_cast<const hipdnnDataType_t*>(arrayOfElements));
}

void getDataType(hipdnn_data_sdk::data_objects::DataType source,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements,
                 const char* errorPrefix)
{
    THROW_IF_FALSE(attributeType == HIPDNN_TYPE_DATA_TYPE,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": attributeType mismatch");
    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   std::string(errorPrefix) + ": requestedElementCount < 1");
    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  std::string(errorPrefix) + ": arrayOfElements is null");
    *static_cast<hipdnnDataType_t*>(arrayOfElements) = fromSdkDataType(source);
    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
}

} // namespace hipdnn_backend

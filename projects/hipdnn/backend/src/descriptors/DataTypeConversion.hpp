// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnBackendAttributeType.h"
#include "HipdnnDataType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>

namespace hipdnn_backend
{

// Converts between C-API hipdnnDataType_t and SDK DataType enum values.
hipdnn_data_sdk::data_objects::DataType toSdkDataType(hipdnnDataType_t type);
hipdnnDataType_t fromSdkDataType(hipdnn_data_sdk::data_objects::DataType type);

// Returns the byte size for a given data type, or -1 if unsupported.
int64_t getDataTypeByteSize(hipdnn_data_sdk::data_objects::DataType type);

// Validates attributeType/elementCount/null and sets a DataType from a
// hipdnnDataType_t value. Mirrors the setScalar/getScalar pattern in
// DescriptorAttributeUtils.hpp.
void setDataType(hipdnn_data_sdk::data_objects::DataType& target,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 const void* arrayOfElements,
                 const char* errorPrefix);

void getDataType(hipdnn_data_sdk::data_objects::DataType source,
                 hipdnnBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements,
                 const char* errorPrefix);

} // namespace hipdnn_backend

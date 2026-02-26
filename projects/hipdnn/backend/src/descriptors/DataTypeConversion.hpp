// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnDataType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>

namespace hipdnn_backend
{

// Converts between C-API hipdnnDataType_t and SDK DataType enum values.
hipdnn_data_sdk::data_objects::DataType toSdkDataType(hipdnnDataType_t type);
hipdnnDataType_t fromSdkDataType(hipdnn_data_sdk::data_objects::DataType type);

// Returns the byte size for a given data type. Throws for unsupported types.
int64_t getDataTypeByteSize(hipdnn_data_sdk::data_objects::DataType type);

} // namespace hipdnn_backend

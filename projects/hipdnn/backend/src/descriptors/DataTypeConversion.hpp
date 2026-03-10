// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnConvolutionMode.h"
#include "HipdnnDataType.h"
#include "HipdnnException.hpp"
#include "HipdnnNormFwdPhase.h"
#include "HipdnnPointwiseMode.h"
#include <hipdnn_data_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
#include <hipdnn_data_sdk/data_objects/norm_common_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>

namespace hipdnn_backend
{

// Converts between C-API hipdnnDataType_t and SDK DataType enum values.
hipdnn_data_sdk::data_objects::DataType toSdkDataType(hipdnnDataType_t type);
hipdnnDataType_t fromSdkDataType(hipdnn_data_sdk::data_objects::DataType type);

// Returns the byte size for a given data type. Throws for unsupported types.
int64_t getDataTypeByteSize(hipdnn_data_sdk::data_objects::DataType type);

// Converts between C-API hipdnnConvolutionMode_t and SDK ConvMode enum values.
hipdnn_data_sdk::data_objects::ConvMode toSdkConvMode(hipdnnConvolutionMode_t mode);
hipdnnConvolutionMode_t fromSdkConvMode(hipdnn_data_sdk::data_objects::ConvMode mode);

// Converts between C-API hipdnnPointwiseMode_t and SDK PointwiseMode enum values.
hipdnn_data_sdk::data_objects::PointwiseMode toSdkPointwiseMode(hipdnnPointwiseMode_t mode);
hipdnnPointwiseMode_t fromSdkPointwiseMode(hipdnn_data_sdk::data_objects::PointwiseMode mode);

// Converts between C-API hipdnnNormFwdPhase_t and SDK NormFwdPhase enum values.
hipdnn_data_sdk::data_objects::NormFwdPhase toSdkNormFwdPhase(hipdnnNormFwdPhase_t phase);
hipdnnNormFwdPhase_t fromSdkNormFwdPhase(hipdnn_data_sdk::data_objects::NormFwdPhase phase);

} // namespace hipdnn_backend

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnAttentionImplementation.h"
#include "HipdnnConvolutionMode.h"
#include "HipdnnDataType.h"
#include "HipdnnDiagonalAlignment.h"
#include "HipdnnException.hpp"
#include "HipdnnNormFwdPhase.h"
#include "HipdnnPaddingMode.h"
#include "HipdnnPointwiseMode.h"
#include "HipdnnReduceTensorOp.h"
#include "HipdnnResampleMode.h"
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/norm_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>

namespace hipdnn_backend
{

// Converts between C-API hipdnnDataType_t and SDK DataType enum values.
hipdnn_flatbuffers_sdk::data_objects::DataType toSdkDataType(hipdnnDataType_t type);
hipdnnDataType_t fromSdkDataType(hipdnn_flatbuffers_sdk::data_objects::DataType type);

// Returns the byte size for a given data type. Throws for unsupported types.
int64_t getDataTypeByteSize(hipdnn_flatbuffers_sdk::data_objects::DataType type);

// Converts between C-API hipdnnConvolutionMode_t and SDK ConvMode enum values.
hipdnn_flatbuffers_sdk::data_objects::ConvMode toSdkConvMode(hipdnnConvolutionMode_t mode);
hipdnnConvolutionMode_t fromSdkConvMode(hipdnn_flatbuffers_sdk::data_objects::ConvMode mode);

// Converts between C-API hipdnnPointwiseMode_t and SDK PointwiseMode enum values.
hipdnn_flatbuffers_sdk::data_objects::PointwiseMode toSdkPointwiseMode(hipdnnPointwiseMode_t mode);
hipdnnPointwiseMode_t
    fromSdkPointwiseMode(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode);

// Converts between C-API hipdnnDiagonalAlignment_t and SDK DiagonalAlignment enum values.
hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment
    toSdkDiagonalAlignment(hipdnnDiagonalAlignment_t mode);
hipdnnDiagonalAlignment_t
    fromSdkDiagonalAlignment(hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment mode);

// Converts between C-API hipdnnAttentionImplementation_t and SDK AttentionImplementation enum
// values.
hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation
    toSdkAttentionImplementation(hipdnnAttentionImplementation_t mode);
hipdnnAttentionImplementation_t fromSdkAttentionImplementation(
    hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation mode);

// Converts between C-API hipdnnNormFwdPhase_t and SDK NormFwdPhase enum values.
hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase toSdkNormFwdPhase(hipdnnNormFwdPhase_t phase);
hipdnnNormFwdPhase_t fromSdkNormFwdPhase(hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase phase);

// Converts between C-API hipdnnReduceTensorOp_t and SDK ReductionMode enum values.
hipdnn_flatbuffers_sdk::data_objects::ReductionMode toSdkReductionMode(hipdnnReduceTensorOp_t mode);
hipdnnReduceTensorOp_t
    fromSdkReductionMode(hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode);

// Converts between C-API hipdnnResampleMode_t and SDK ResampleMode enum values.
hipdnn_flatbuffers_sdk::data_objects::ResampleMode toSdkResampleMode(hipdnnResampleMode_t mode);
hipdnnResampleMode_t fromSdkResampleMode(hipdnn_flatbuffers_sdk::data_objects::ResampleMode mode);

// Converts between C-API hipdnnPaddingMode_t and SDK PaddingMode enum values.
hipdnn_flatbuffers_sdk::data_objects::PaddingMode toSdkPaddingMode(hipdnnPaddingMode_t mode);
hipdnnPaddingMode_t fromSdkPaddingMode(hipdnn_flatbuffers_sdk::data_objects::PaddingMode mode);

} // namespace hipdnn_backend

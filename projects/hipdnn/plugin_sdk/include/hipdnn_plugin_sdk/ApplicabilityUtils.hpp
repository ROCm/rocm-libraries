// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>

/// Report the enclosing isApplicable as not applicable when the graph contains an
/// unsupported ragged tensor, by returning false from the caller.
///
/// @param tensor_map  UID-to-TensorAttributes map from the op-graph under evaluation.
#define CHECK_NO_RAGGED_TENSORS(tensor_map)                                        \
    do                                                                             \
    {                                                                              \
        if(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds((tensor_map))) \
        {                                                                          \
            return false;                                                          \
        }                                                                          \
    } while(0)

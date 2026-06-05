// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnReduceTensorOp.h
 * @brief Reduce tensor operator enumeration for hipDNN backend operations
 *
 * Defines the hipdnnReduceTensorOp_t used when setting the
 * HIPDNN_ATTR_REDUCTION_OPERATOR attribute on Reduction descriptors.
 */

#pragma once

/**
 * @enum hipdnnReduceTensorOp_t
 * @brief Reduce tensor operator for backend Reduction operations
 */
typedef enum
{
    HIPDNN_REDUCE_TENSOR_ADD = 1, ///< Sum reduction
    HIPDNN_REDUCE_TENSOR_MUL = 2, ///< Product reduction
    HIPDNN_REDUCE_TENSOR_MIN = 3, ///< Minimum reduction
    HIPDNN_REDUCE_TENSOR_MAX = 4, ///< Maximum reduction
    HIPDNN_REDUCE_TENSOR_AMAX = 5, ///< Absolute maximum reduction
    HIPDNN_REDUCE_TENSOR_AVG = 6, ///< Average reduction
    HIPDNN_REDUCE_TENSOR_NORM1 = 7, ///< L1 norm reduction
    HIPDNN_REDUCE_TENSOR_NORM2 = 8, ///< L2 norm reduction
    HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 9 ///< Product reduction excluding zeros
} hipdnnReduceTensorOp_t;

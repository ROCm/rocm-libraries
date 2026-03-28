// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnReductionMode.h
 * @brief ReductionMode enumeration for hipDNN backend operations
 *
 * Defines the ReductionMode used when setting the
 * HIPDNN_ATTR_REDUCTION_MODE_EXT attribute on Reduction descriptors.
 */

#pragma once

/**
 * @enum hipdnnReductionMode_t
 * @brief ReductionMode for backend Reduction operations
 */
typedef enum
{
    HIPDNN_REDUCTION_ADD = 1, ///< Sum reduction
    HIPDNN_REDUCTION_MUL = 2, ///< Product reduction
    HIPDNN_REDUCTION_MIN = 3, ///< Minimum reduction
    HIPDNN_REDUCTION_MAX = 4, ///< Maximum reduction
    HIPDNN_REDUCTION_AMAX = 5, ///< Absolute maximum reduction
    HIPDNN_REDUCTION_AVG = 6, ///< Average reduction
    HIPDNN_REDUCTION_NORM1 = 7, ///< L1 norm reduction
    HIPDNN_REDUCTION_NORM2 = 8, ///< L2 norm reduction
    HIPDNN_REDUCTION_MUL_NO_ZEROS = 9 ///< Product reduction excluding zeros
} hipdnnReductionMode_t;

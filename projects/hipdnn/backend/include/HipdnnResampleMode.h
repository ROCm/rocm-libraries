// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnResampleMode.h
 * @brief ResampleMode enumeration for hipDNN backend operations
 *
 * Defines the ResampleMode used when setting the
 * HIPDNN_ATTR_RESAMPLE_MODE attribute on ResampleFwd descriptors.
 */

#pragma once

/**
 * @enum hipdnnResampleMode_t
 * @brief ResampleMode for backend ResampleFwd operations
 */
typedef enum
{
    HIPDNN_RESAMPLE_MAXPOOL = 1, ///< Maximum resample
    HIPDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING = 2, ///< Average resample (excludes padding)
    HIPDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING = 3 ///< Average resample (includes padding)
} hipdnnResampleMode_t;

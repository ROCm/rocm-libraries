// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnPoolingMode.h
 * @brief PoolingMode enumeration for hipDNN backend operations
 *
 * Defines the PoolingMode used when setting the
 * HIPDNN_ATTR_POOLING_MODE attribute on PoolingFwd descriptors.
 */

#pragma once

/**
 * @enum hipdnnPoolingMode_t
 * @brief PoolingMode for backend PoolingFwd operations
 */
typedef enum
{
    HIPDNN_POOLING_MODE_MAX = 1, ///< Maximum pooling
    HIPDNN_POOLING_MODE_AVERAGE = 2, ///< Average pooling (excludes padding)
    HIPDNN_POOLING_MODE_AVERAGE_INCLUSIVE = 3 ///< Average pooling (includes padding)
} hipdnnPoolingMode_t;

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file HipdnnDataType.h
 * @brief Data type identifiers for tensor element types
 */
#pragma once

/**
 * @enum hipdnnDataType_t
 * @brief Tensor element data types
 */
typedef enum
{
    HIPDNN_DATA_FLOAT = 0, ///< 32-bit floating point
    HIPDNN_DATA_DOUBLE = 1, ///< 64-bit floating point
    HIPDNN_DATA_HALF = 2, ///< 16-bit floating point (IEEE 754)
    HIPDNN_DATA_INT8 = 3, ///< 8-bit signed integer
    HIPDNN_DATA_INT32 = 4, ///< 32-bit signed integer
    HIPDNN_DATA_UINT8 = 5, ///< 8-bit unsigned integer
    HIPDNN_DATA_BFLOAT16 = 6, ///< 16-bit brain floating point
    HIPDNN_DATA_FP8_E4M3 = 7, ///< 8-bit floating point (E4M3)
    HIPDNN_DATA_FP8_E5M2 = 8, ///< 8-bit floating point (E5M2)
    HIPDNN_DATA_FP8_E8M0 = 9, ///< 8-bit floating point (E8M0)
    HIPDNN_DATA_FP4_E2M1 = 10, ///< 4-bit floating point (E2M1)
    HIPDNN_DATA_INT4 = 11, ///< 4-bit signed integer
    HIPDNN_DATA_FP6_E2M3_EXT = 12, ///< 6-bit floating point (E2M3)
    HIPDNN_DATA_FP6_E3M2_EXT = 13, ///< 6-bit floating point (E3M2)
    HIPDNN_DATA_INT64 = 14, ///< 64-bit signed integer
    HIPDNN_DATA_BOOLEAN = 15, ///< 8-bit boolean
} hipdnnDataType_t;

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnPaddingMode.h
 * @brief PaddingMode enumeration for hipDNN backend operations
 *
 * Defines the PaddingMode used when setting the
 * HIPDNN_ATTR_RESAMPLE_PADDING_MODE attribute on ResampleFwd descriptors.
 */

#pragma once

/**
 * @enum hipdnnPaddingMode_t
 * @brief PaddingMode for backend ResampleFwd operations
 */
typedef enum
{
    HIPDNN_PADDING_NEG_INF_PAD = 1, ///< Pad with negative infinity
    HIPDNN_PADDING_ZERO_PAD = 2 ///< Pad with zeros
} hipdnnPaddingMode_t;

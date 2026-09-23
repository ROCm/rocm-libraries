// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnDiagonalAlignment.h
 * @brief Diagonal alignment mode enumeration for hipDNN SDPA operations
 *
 * Defines the diagonal alignment mode used when setting the
 * HIPDNN_ATTR_SDPA_FWD_DIAGONAL_ALIGNMENT_EXT attribute on SDPA descriptors.
 */

#pragma once

/**
 * @enum hipdnnDiagonalAlignment_t
 * @brief Diagonal alignment mode for SDPA attention mask
 *
 * Determines which corner of the attention matrix the causal mask diagonal
 * is aligned to.
 *
 * @see HIPDNN_ATTR_SDPA_FWD_DIAGONAL_ALIGNMENT_EXT
 * @see hipdnnBackendSetAttribute()
 */
typedef enum
{
    HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT = 0,
    HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT = 1
} hipdnnDiagonalAlignment_t;

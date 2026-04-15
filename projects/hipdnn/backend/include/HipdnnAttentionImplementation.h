// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnAttentionImplementation.h
 * @brief Attention implementation mode enumeration for hipDNN SDPA operations
 *
 * Defines the attention implementation mode used when setting the
 * HIPDNN_ATTR_SDPA_FWD_IMPLEMENTATION_EXT attribute on SDPA descriptors.
 */

#pragma once

/**
 * @enum hipdnnAttentionImplementation_t
 * @brief Attention implementation strategy for SDPA operations
 *
 * Determines which implementation strategy to use for the scaled dot-product
 * attention computation.
 *
 * @see HIPDNN_ATTR_SDPA_FWD_IMPLEMENTATION_EXT
 * @see hipdnnBackendSetAttribute()
 */
typedef enum
{
    HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT = 0,
    HIPDNN_ATTENTION_IMPLEMENTATION_COMPOSITE_EXT = 1,
    HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT = 2
} hipdnnAttentionImplementation_t;

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnMoeGroupedMatmulMode.h
 * @brief MoeGroupedMatmulMode enumeration for hipDNN backend operations
 *
 * Defines the MoeGroupedMatmulMode used when setting the
 * HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE attribute on MoeGroupedMatmul descriptors.
 */

#pragma once

/**
 * @enum hipdnnMoeGroupedMatmulMode_t
 * @brief MoeGroupedMatmulMode for backend MoeGroupedMatmul operations
 */
typedef enum
{
    HIPDNN_MOE_GROUPED_MATMUL_MODE_NONE = 0, ///< Tokens are already routed
    HIPDNN_MOE_GROUPED_MATMUL_MODE_GATHER = 1, ///< Gather tokens before grouped matmul
    HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER
    = 2 ///< Scatter grouped matmul output to source token order
} hipdnnMoeGroupedMatmulMode_t;

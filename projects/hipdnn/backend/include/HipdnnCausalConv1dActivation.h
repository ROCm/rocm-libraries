// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnCausalConv1dActivation.h
 * @brief CausalConv1dActivation enumeration for hipDNN backend operations
 *
 * Defines the CausalConv1dActivation used when setting the
 * HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT attribute on CausalConv1d descriptors.
 */

#pragma once

/**
 * @enum hipdnnCausalConv1dActivation_t
 * @brief CausalConv1dActivation for backend CausalConv1d operations
 */
typedef enum
{
    HIPDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY_EXT = 0, ///< No activation
    HIPDNN_CAUSAL_CONV1D_ACTIVATION_SILU_EXT = 1 ///< SiLU activation, x * sigmoid(x)
} hipdnnCausalConv1dActivation_t;

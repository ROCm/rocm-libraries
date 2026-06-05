// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnConvolutionMode.h
 * @brief Convolution mode enumeration for hipDNN backend operations
 *
 * Defines the convolution mode used when setting the
 * HIPDNN_ATTR_CONVOLUTION_CONV_MODE attribute on convolution descriptors.
 */

#pragma once

/**
 * @enum hipdnnConvolutionMode_t
 * @brief Convolution mode for backend convolution operations
 *
 * Determines how the convolution filter is applied to the input tensor.
 */
typedef enum
{
    HIPDNN_CONVOLUTION = 0, ///< Mathematical convolution (filter is flipped)
    HIPDNN_CROSS_CORRELATION = 1 ///< Cross-correlation mode (standard deep learning)
} hipdnnConvolutionMode_t;

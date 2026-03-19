// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnOperationType.h
 * @brief Operation type enumeration for hipDNN backend operation descriptors
 *
 * Defines the operation type used when querying the
 * HIPDNN_ATTR_OPERATION_TYPE_EXT attribute on operation descriptors.
 */

#pragma once

/**
 * @enum hipdnnOperationType_t
 * @brief Operation type for backend operation descriptors
 *
 * Identifies the type of a backend operation descriptor, enabling
 * type-based dispatch without trial-and-error attribute probing.
 */
typedef enum
{
    HIPDNN_OPERATION_TYPE_NOT_SET = 0, ///< Operation type not specified
    HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD = 1, ///< Convolution forward pass
    HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_DATA = 2, ///< Convolution backward data pass
    HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_WEIGHTS = 3, ///< Convolution backward weights pass
    HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE = 4, ///< Batch normalization inference
    HIPDNN_OPERATION_TYPE_BATCHNORM_BACKWARD = 5, ///< Batch normalization backward pass
    HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_VARIANCE
    = 6, ///< Batch normalization inference with variance
    HIPDNN_OPERATION_TYPE_BATCHNORM = 7, ///< Batch normalization training forward
    HIPDNN_OPERATION_TYPE_POINTWISE = 8, ///< Pointwise operation
    HIPDNN_OPERATION_TYPE_MATMUL = 9, ///< Matrix multiplication
    HIPDNN_OPERATION_TYPE_RMSNORM = 10, ///< RMS normalization
    HIPDNN_OPERATION_TYPE_LAYERNORM = 11, ///< Layer normalization
    HIPDNN_OPERATION_TYPE_SDPA_FORWARD = 12, ///< Scaled dot-product attention forward
    HIPDNN_OPERATION_TYPE_BLOCK_SCALE_QUANTIZE = 13, ///< Block scale quantization
    HIPDNN_OPERATION_TYPE_SDPA_BACKWARD = 14, ///< Scaled dot-product attention backward
    HIPDNN_OPERATION_TYPE_BLOCK_SCALE_DEQUANTIZE = 15, ///< Block scale dequantization
    HIPDNN_OPERATION_TYPE_CUSTOM_OP = 16, ///< Custom operation
    HIPDNN_OPERATION_TYPE_REDUCTION = 17, ///< Reduction operation
} hipdnnOperationType_t;

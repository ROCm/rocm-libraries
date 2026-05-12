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
 * @enum hipdnnOperationType_ext_t
 * @brief Operation type for backend operation descriptors
 *
 * Identifies the type of a backend operation descriptor, enabling
 * type-based dispatch without trial-and-error attribute probing.
 */
typedef enum
{
    HIPDNN_OPERATION_TYPE_NOT_SET_EXT = 0, ///< Operation type not specified
    HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT = 1, ///< Convolution forward pass
    HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_DATA_EXT = 2, ///< Convolution backward data pass
    HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_WEIGHTS_EXT
    = 3, ///< Convolution backward weights pass
    HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_EXT = 4, ///< Batch normalization inference
    HIPDNN_OPERATION_TYPE_BATCHNORM_BACKWARD_EXT = 5, ///< Batch normalization backward pass
    HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_VARIANCE_EXT
    = 6, ///< Batch normalization inference with variance
    HIPDNN_OPERATION_TYPE_BATCHNORM_EXT = 7, ///< Batch normalization training forward
    HIPDNN_OPERATION_TYPE_POINTWISE_EXT = 8, ///< Pointwise operation
    HIPDNN_OPERATION_TYPE_MATMUL_EXT = 9, ///< Matrix multiplication
    HIPDNN_OPERATION_TYPE_RMSNORM_EXT = 10, ///< RMS normalization
    HIPDNN_OPERATION_TYPE_LAYERNORM_EXT = 11, ///< Layer normalization
    HIPDNN_OPERATION_TYPE_SDPA_FORWARD_EXT = 12, ///< Scaled dot-product attention forward
    HIPDNN_OPERATION_TYPE_BLOCK_SCALE_QUANTIZE_EXT = 13, ///< Block scale quantization
    HIPDNN_OPERATION_TYPE_SDPA_BACKWARD_EXT = 14, ///< Scaled dot-product attention backward
    HIPDNN_OPERATION_TYPE_BLOCK_SCALE_DEQUANTIZE_EXT = 15, ///< Block scale dequantization
    HIPDNN_OPERATION_TYPE_CUSTOM_OP_EXT = 16, ///< Custom operation
    HIPDNN_OPERATION_TYPE_REDUCTION_EXT = 17, ///< Reduction operation
    HIPDNN_OPERATION_TYPE_RESAMPLE_FWD = 18, ///< Resample forward operation
} hipdnnOperationType_ext_t;

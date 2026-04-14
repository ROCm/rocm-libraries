// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnBackendAttributeType.h
 * @brief Data type identifiers for hipDNN backend attribute values
 *
 * This file defines the types used to specify the data type of attribute
 * values when calling hipdnnBackendSetAttribute() and hipdnnBackendGetAttribute().
 */

#pragma once

/**
 * @enum hipdnnBackendAttributeType_t
 * @brief Data types for backend descriptor attribute values
 *
 * When setting or getting attributes on backend descriptors, you must specify
 * the data type of the value being passed. These constants identify the
 * expected type for proper marshalling.
 */
typedef enum
{
    /** @brief hipDNN handle (hipdnnHandle_t) */
    HIPDNN_TYPE_HANDLE = 0,

    /** @brief Data type enumeration (hipdnnDataType_t) */
    HIPDNN_TYPE_DATA_TYPE = 1,

    /** @brief Boolean value (bool) */
    HIPDNN_TYPE_BOOLEAN = 2,

    /** @brief 64-bit signed integer (int64_t) */
    HIPDNN_TYPE_INT64 = 3,

    /** @brief Single precision floating point (float) */
    HIPDNN_TYPE_FLOAT = 4,

    /** @brief Double precision floating point (double) */
    HIPDNN_TYPE_DOUBLE = 5,

    /** @brief Generic pointer (void*) - typically device memory */
    HIPDNN_TYPE_VOID_PTR = 6,

    /** @brief Heuristic mode enumeration (hipdnnBackendHeuristicMode_t) */
    HIPDNN_TYPE_HEUR_MODE = 7,

    /** @brief Knob type identifier */
    HIPDNN_TYPE_KNOB_TYPE = 8,

    /** @brief NaN propagation mode */
    HIPDNN_TYPE_NAN_PROPOGATION = 9,

    /** @brief Numerical precision/behavior note */
    HIPDNN_TYPE_NUMERICAL_NOTE = 10,

    /** @brief Tensor layout type */
    HIPDNN_TYPE_LAYOUT_TYPE = 11,

    /** @brief Attribute name enumeration (hipdnnBackendAttributeName_t) */
    HIPDNN_TYPE_ATTRIB_NAME = 12,

    /** @brief Backend descriptor handle (hipdnnBackendDescriptor_t) */
    HIPDNN_TYPE_BACKEND_DESCRIPTOR = 13,

    /** @brief Generate statistics mode */
    HIPDNN_TYPE_GENSTATS_MODE = 14,

    /** @brief Batch normalization finalize statistics mode */
    HIPDNN_TYPE_BN_FINALIZE_STATS_MODE = 15,

    /** @brief Engine behavior note */
    HIPDNN_TYPE_BEHAVIOR_NOTE = 16,

    /** @brief Tensor reordering mode */
    HIPDNN_TYPE_TENSOR_REORDERING_MODE = 17,

    /** @brief 32-bit signed integer (int32_t) */
    HIPDNN_TYPE_INT32 = 18,

    /** @brief Character (char) */
    HIPDNN_TYPE_CHAR = 19,

    /** @brief Signal mode for synchronization */
    HIPDNN_TYPE_SIGNAL_MODE = 20,

    /** @brief Fractional value type */
    HIPDNN_TYPE_FRACTION = 21,

    /** @brief Normalization forward phase */
    HIPDNN_TYPE_NORM_FWD_PHASE = 22,

    /** @brief Random number generator distribution */
    HIPDNN_TYPE_RNG_DISTRIBUTION = 23,

    /** @brief Convolution mode enumeration (hipdnnConvolutionMode_t) */
    HIPDNN_TYPE_CONVOLUTION_MODE = 24,

    /** @brief Pointwise mode enumeration (hipdnnPointwiseMode_t) */
    HIPDNN_TYPE_POINTWISE_MODE = 25,

    /** @brief Diagonal alignment mode enumeration (hipdnnDiagonalAlignment_t) */
    HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT = 26,

    /** @brief Attention implementation mode enumeration (hipdnnAttentionImplementation_t) */
    HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT = 27,

    /** @brief Reduce tensor operator enumeration (hipdnnReduceTensorOp_t) */
    HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE = 28,

    /**
     * @name Extension Types
     * hipDNN-specific extension types
     * @{
     */

    /**
     * @brief Serialized FlatBuffer data structure (extension)
     *
     * Used for passing serialized FlatBuffer data to/from the backend.
     * The value should be a hipdnnBackendFlatbufferData_t struct.
     */
    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT = 10000,

    /**
     * @brief Operation type enumeration (hipdnnOperationType_ext_t, extension)
     *
     * Used for querying the operation type of an operation descriptor
     * via HIPDNN_ATTR_OPERATION_TYPE_EXT.
     */
    HIPDNN_TYPE_OPERATION_TYPE_EXT = 10001

    /** @} */

} hipdnnBackendAttributeType_t;

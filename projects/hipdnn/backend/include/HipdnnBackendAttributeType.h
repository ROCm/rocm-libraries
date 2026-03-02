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
    HIPDNN_TYPE_DATA_TYPE,

    /** @brief Boolean value (bool) */
    HIPDNN_TYPE_BOOLEAN,

    /** @brief 64-bit signed integer (int64_t) */
    HIPDNN_TYPE_INT64,

    /** @brief Single precision floating point (float) */
    HIPDNN_TYPE_FLOAT,

    /** @brief Double precision floating point (double) */
    HIPDNN_TYPE_DOUBLE,

    /** @brief Generic pointer (void*) - typically device memory */
    HIPDNN_TYPE_VOID_PTR,

    /** @brief Heuristic mode enumeration (hipdnnBackendHeuristicMode_t) */
    HIPDNN_TYPE_HEUR_MODE,

    /** @brief Knob type identifier */
    HIPDNN_TYPE_KNOB_TYPE,

    /** @brief NaN propagation mode */
    HIPDNN_TYPE_NAN_PROPOGATION,

    /** @brief Numerical precision/behavior note */
    HIPDNN_TYPE_NUMERICAL_NOTE,

    /** @brief Tensor layout type */
    HIPDNN_TYPE_LAYOUT_TYPE,

    /** @brief Attribute name enumeration (hipdnnBackendAttributeName_t) */
    HIPDNN_TYPE_ATTRIB_NAME,

    /** @brief Backend descriptor handle (hipdnnBackendDescriptor_t) */
    HIPDNN_TYPE_BACKEND_DESCRIPTOR,

    /** @brief Generate statistics mode */
    HIPDNN_TYPE_GENSTATS_MODE,

    /** @brief Batch normalization finalize statistics mode */
    HIPDNN_TYPE_BN_FINALIZE_STATS_MODE,

    /** @brief Engine behavior note */
    HIPDNN_TYPE_BEHAVIOR_NOTE,

    /** @brief Tensor reordering mode */
    HIPDNN_TYPE_TENSOR_REORDERING_MODE,

    /** @brief 32-bit signed integer (int32_t) */
    HIPDNN_TYPE_INT32,

    /** @brief Character (char) */
    HIPDNN_TYPE_CHAR,

    /** @brief Signal mode for synchronization */
    HIPDNN_TYPE_SIGNAL_MODE,

    /** @brief Fractional value type */
    HIPDNN_TYPE_FRACTION,

    /** @brief Normalization forward phase */
    HIPDNN_TYPE_NORM_FWD_PHASE,

    /** @brief Random number generator distribution */
    HIPDNN_TYPE_RNG_DISTRIBUTION,

    /** @brief Convolution mode enumeration (hipdnnConvolutionMode_t) */
    HIPDNN_TYPE_CONVOLUTION_MODE,

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
    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT

    /** @} */

} hipdnnBackendAttributeType_t;

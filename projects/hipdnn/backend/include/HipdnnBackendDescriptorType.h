// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnBackendDescriptorType.h
 * @brief Descriptor type identifiers for hipDNN backend API
 *
 * This file defines the types of backend descriptors that can be created
 * using hipdnnBackendCreateDescriptor(). Each descriptor type holds specific
 * configuration for different aspects of graph execution.
 */

#pragma once

/**
 * @enum hipdnnBackendDescriptorType_t
 * @brief Types of backend descriptors
 *
 * Backend descriptors are opaque handles that store configuration data.
 * Use hipdnnBackendCreateDescriptor() with one of these types to create
 * a new descriptor, then configure it with hipdnnBackendSetAttribute().
 *
 * @see hipdnnBackendCreateDescriptor()
 * @see hipdnnBackendSetAttribute()
 * @see hipdnnBackendFinalize()
 */
typedef enum
{
    /** @brief Invalid descriptor type - used for error detection */
    HIPDNN_INVALID_TYPE = 0,

    /**
     * @brief Engine descriptor
     *
     * Represents a specific execution engine that can run an operation graph.
     * Engines are discovered through heuristics and provide different
     * performance/precision trade-offs.
     */
    HIPDNN_BACKEND_ENGINE_DESCRIPTOR,

    /**
     * @brief Engine configuration descriptor
     *
     * Holds the configuration for an engine, including knob settings
     * and workspace requirements. Created from an engine descriptor.
     */
    HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR,

    /**
     * @brief Engine heuristic descriptor
     *
     * Used to query available engines for an operation graph.
     * Set the operation graph and heuristic mode, then query results
     * to get ranked engine configurations.
     */
    HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,

    /**
     * @brief Execution plan descriptor
     *
     * The final compiled plan ready for execution. Created from an
     * engine configuration and used with hipdnnBackendExecute().
     */
    HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,

    /**
     * @brief Intermediate tensor info descriptor
     *
     * Contains information about intermediate (virtual) tensors
     * in a fused operation graph.
     */
    HIPDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR,

    /**
     * @brief Knob choice descriptor
     *
     * Represents a specific value for an engine tuning knob.
     * Used to configure engine behavior.
     */
    HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR,

    /**
     * @brief Knob info descriptor
     *
     * Contains metadata about an engine knob, including
     * valid value ranges and default values.
     */
    HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR,

    /**
     * @brief Layout info descriptor
     *
     * Contains information about tensor layout requirements
     * and supported formats for an engine.
     */
    HIPDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR,

    /**
     * @brief Generate statistics operation descriptor
     *
     * Represents an operation that generates statistics (e.g., for
     * batch normalization running mean/variance).
     */
    HIPDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,

    /**
     * @brief Operation graph descriptor
     *
     * The main descriptor representing a computational graph.
     * Contains tensors and operations to be executed.
     * Typically created from serialized FlatBuffer data.
     */
    HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,

    /**
     * @brief Variant pack descriptor
     *
     * Maps tensor UIDs to device memory pointers and workspace
     * for execution. Passed to hipdnnBackendExecute().
     */
    HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,

    /**
     * @brief Kernel cache descriptor
     *
     * Manages caching of compiled GPU kernels for faster
     * subsequent execution plan creation.
     */
    HIPDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR,

    /**
     * @brief Paged cache load operation descriptor
     *
     * Represents a paged memory cache loading operation.
     */
    HIPDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,

    /**
     * @brief Tensor descriptor
     *
     * Represents a tensor with dimensions, strides, data type,
     * and optional pass-by-value data.
     */
    HIPDNN_BACKEND_TENSOR_DESCRIPTOR,

    /**
     * @brief Convolution forward operation descriptor
     *
     * Represents a forward convolution operation with input (X),
     * weight (W), and output (Y) tensors plus convolution parameters.
     */
    HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,

    /**
     * @brief Convolution backward filter operation descriptor
     *
     * Represents a backward filter convolution operation with input (X),
     * output gradient (DY), and weight gradient (DW) tensors plus
     * convolution parameters.
     */
    HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,

    /**
     * @brief Batch normalization inference operation descriptor
     *
     * Represents a batch normalization inference operation with input (X),
     * mean, inverse variance, scale, bias, and output (Y) tensors.
     */
    HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT,

    /**
     * @brief Convolution backward data (Dgrad) operation descriptor
     *
     * Represents a backward data convolution operation with output gradient (DY),
     * weight (W), and input gradient (DX) tensors plus convolution parameters.
     */
    HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DESCRIPTOR,

} hipdnnBackendDescriptorType_t;

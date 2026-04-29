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
    HIPDNN_INVALID_TYPE_EXT = 0,

    /**
     * @brief Engine descriptor
     *
     * Represents a specific execution engine that can run an operation graph.
     * Engines are discovered through heuristics and provide different
     * performance/precision trade-offs.
     */
    HIPDNN_BACKEND_ENGINE_DESCRIPTOR = 1,

    /**
     * @brief Engine configuration descriptor
     *
     * Holds the configuration for an engine, including knob settings
     * and workspace requirements. Created from an engine descriptor.
     */
    HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR = 2,

    /**
     * @brief Engine heuristic descriptor
     *
     * Used to query available engines for an operation graph.
     * Set the operation graph and heuristic mode, then query results
     * to get ranked engine configurations.
     */
    HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR = 3,

    /**
     * @brief Execution plan descriptor
     *
     * The final compiled plan ready for execution. Created from an
     * engine configuration and used with hipdnnBackendExecute().
     */
    HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR = 4,

    /**
     * @brief Intermediate tensor info descriptor
     *
     * Contains information about intermediate (virtual) tensors
     * in a fused operation graph.
     */
    HIPDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR = 5,

    /**
     * @brief Knob choice descriptor
     *
     * Represents a specific value for an engine tuning knob.
     * Used to configure engine behavior.
     */
    HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR = 6,

    /**
     * @brief Knob info descriptor
     *
     * Contains metadata about an engine knob, including
     * valid value ranges and default values.
     */
    HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR = 7,

    /**
     * @brief Layout info descriptor
     *
     * Contains information about tensor layout requirements
     * and supported formats for an engine.
     */
    HIPDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR = 8,

    /**
     * @brief Generate statistics operation descriptor
     *
     * Represents an operation that generates statistics (e.g., for
     * batch normalization running mean/variance).
     */
    HIPDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = 9,

    /**
     * @brief Operation graph descriptor
     *
     * The main descriptor representing a computational graph.
     * Contains tensors and operations to be executed.
     * Typically created from serialized FlatBuffer data.
     */
    HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR = 10,

    /**
     * @brief Variant pack descriptor
     *
     * Maps tensor UIDs to device memory pointers and workspace
     * for execution. Passed to hipdnnBackendExecute().
     */
    HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = 11,

    /**
     * @brief Kernel cache descriptor
     *
     * Manages caching of compiled GPU kernels for faster
     * subsequent execution plan creation.
     */
    HIPDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR = 12,

    /**
     * @brief Paged cache load operation descriptor
     *
     * Represents a paged memory cache loading operation.
     */
    HIPDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR = 13,

    /**
     * @brief Tensor descriptor
     *
     * Represents a tensor with dimensions, strides, data type,
     * and optional pass-by-value data.
     */
    HIPDNN_BACKEND_TENSOR_DESCRIPTOR = 14,

    /**
     * @brief Convolution forward operation descriptor
     *
     * Represents a forward convolution operation with input (X),
     * weight (W), and output (Y) tensors plus convolution parameters.
     */
    HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = 15,

    /**
     * @brief Convolution backward filter operation descriptor
     *
     * Represents a backward filter convolution operation with input (X),
     * output gradient (DY), and weight gradient (DW) tensors plus
     * convolution parameters.
     */
    HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = 16,

    /**
     * @brief Batch normalization inference operation descriptor
     *
     * Represents a batch normalization inference operation with input (X),
     * mean, inverse variance, scale, bias, and output (Y) tensors.
     */
    HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT = 17,

    /**
     * @brief Pointwise operation descriptor
     *
     * Represents a pointwise (element-wise) operation with 1-3 input tensors
     * and activation parameters. Supports unary, binary, and ternary operations.
     */
    HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = 18,

    /**
     * @brief Convolution backward data (Dgrad) operation descriptor
     *
     * Represents a backward data convolution operation with output gradient (DY),
     * weight (W), and input gradient (DX) tensors plus convolution parameters.
     */
    HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = 19,

    /**
     * @brief Batchnorm backward operation descriptor (extension)
     *
     * Represents a backward batch normalization operation computing gradients
     * with respect to input (DX), scale (DScale), and bias (DBias).
     */
    HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT = 20,

    /**
     * @brief Batchnorm inference variance ext operation descriptor
     *
     * Represents a batch normalization inference operation (variance ext variant)
     * with input (X), mean, variance, scale, bias, epsilon, and output (Y) tensors.
     */
    HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_VARIANCE_DESCRIPTOR_EXT = 21,

    /**
     * @brief Matrix multiplication operation descriptor
     *
     * Represents a matrix multiplication operation with input (A),
     * input (B), and output (C) tensors plus a compute data type.
     */
    HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = 22,

    /**
     * @brief RMSNorm operation descriptor
     *
     * Represents an RMS normalization forward operation with input (X),
     * scale, epsilon, bias (optional), output (Y), and inverse RMS
     * (optional, training only) tensors.
     */
    HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT = 23,

    /**
     * @brief SDPA forward propagation operation descriptor
     *
     * Represents a scaled dot-product attention forward operation with
     * query (Q), key (K), value (V), and output (O) tensors plus
     * attention parameters.
     */
    HIPDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR = 24,

    /**
     * @brief Layer normalization operation descriptor
     *
     * Represents a layer normalization operation with input (X),
     * scale, bias, epsilon tensors, output (Y), and optional
     * mean and inverse variance outputs plus a compute data type.
     */
    HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT = 25,

    /**
     * @brief Block scale quantize operation descriptor
     *
     * Represents a block scale quantize operation with input (X),
     * output (Y), and scale tensors plus block_size, axis, and transpose parameters.
     */
    HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR = 26,

    /**
     * @brief Batchnorm training forward operation descriptor (extension)
     *
     * Represents a batch normalization training forward operation with
     * input (X), scale, bias, epsilon, output (Y), optional mean and
     * inverse variance outputs, optional running statistics, and peer stats.
     */
    HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT = 27,

    /**
     * @brief Block scale dequantize operation descriptor
     *
     * Represents a block scale dequantize operation with input (X),
     * scale, and output (Y) tensors plus block size parameters.
     */
    HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR = 28,

    /**
     * @brief Custom operation descriptor
     *
     * Represents an opaque custom (plugin-provided) operation with
     * variable-length input/output tensor arrays, a plugin identifier
     * string, and an opaque byte payload interpreted by the plugin.
     */
    HIPDNN_BACKEND_OPERATION_CUSTOM_OP_DESCRIPTOR_EXT = 29,

    /**
     * @brief SDPA backward propagation operation descriptor (extension)
     *
     * Represents a scaled dot-product attention backward operation with
     * query (Q), key (K), value (V), output (O), gradient output (dO),
     * and stats tensors as inputs, producing gradients dQ, dK, dV.
     */
    HIPDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR_EXT = 30,

    /**
     * @brief Reduction operation descriptor
     *
     * Represents a reduction operation with input (X) and output (Y)
     * tensors, a reduction operator, and a compute data type.
     */
    HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = 31,

    /**
     * @brief Resample forward operation descriptor
     *
     * Represents a resample forward operation (max, average, etc.).
     * Takes an input tensor X and produces an output tensor Y,
     * with optional index tensor IDX for max resample.
     */
    HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = 32,

} hipdnnBackendDescriptorType_t;

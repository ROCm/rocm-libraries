// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnBackendAttributeName.h
 * @brief Attribute identifiers for hipDNN backend descriptors
 *
 * This file defines the attribute names used with hipdnnBackendSetAttribute()
 * and hipdnnBackendGetAttribute() to configure and query backend descriptors.
 *
 * Attributes are organized by descriptor type, with each group having a
 * distinct numeric range for identification.
 */

#pragma once

/**
 * @enum hipdnnBackendAttributeName_t
 * @brief Identifiers for backend descriptor attributes
 *
 * These constants are used with hipdnnBackendSetAttribute() and
 * hipdnnBackendGetAttribute() to specify which attribute to access.
 *
 * Attribute ranges by descriptor type:
 * - 100-199: Engine heuristic attributes
 * - 200-299: Engine configuration attributes
 * - 300-399: Execution plan attributes
 * - 400-499: Intermediate info attributes
 * - 500-599: Knob choice attributes
 * - 600-699: Operation graph attributes
 * - 700-799: Variant pack attributes
 * - 800-899: Layout info attributes
 * - 900-999: Knob info attributes
 * - 1000-1099: Engine attributes
 * - 1100-1199: Kernel cache attributes
 * - 1200-1299: Device properties attributes
 * - 60000+: Extension attributes
 */
typedef enum
{
    /**
     * @name Engine Heuristic Attributes (100-199)
     * Attributes for HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR
     * @{
     */

    /** @brief Heuristic search mode (hipdnnBackendHeuristicMode_t) */
    HIPDNN_ATTR_ENGINEHEUR_MODE = 100,

    /** @brief Operation graph to find engines for (backend descriptor) */
    HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 101,

    /** @brief Results from heuristic search (array of engine configs) */
    HIPDNN_ATTR_ENGINEHEUR_RESULTS = 102,

    /** @brief Target SM count for heuristic evaluation */
    HIPDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET = 103,

    /** @brief Device properties for heuristic evaluation */
    HIPDNN_ATTR_ENGINEHEUR_DEVICEPROP = 104,

    /** @} */

    /**
     * @name Engine Configuration Attributes (200-299)
     * Attributes for HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR
     * @{
     */

    /** @brief The engine this configuration is for (backend descriptor) */
    HIPDNN_ATTR_ENGINECFG_ENGINE = 200,

    /** @brief Intermediate tensor information */
    HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO = 201,

    /** @brief Knob choices for this configuration */
    HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES = 202,

    /** @brief Required workspace size in bytes */
    HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE = 203,

    /** @} */

    /**
     * @name Execution Plan Attributes (300-399)
     * Attributes for HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
     * @{
     */

    /** @brief Associated handle (deprecated, do not use) */
    HIPDNN_ATTR_EXECUTION_PLAN_HANDLE = 300,

    /** @brief Engine configuration for this plan (backend descriptor) */
    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = 301,

    /** @brief Required workspace size in bytes (int64_t) */
    HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = 302,

    /** @brief UIDs of computed intermediate tensors */
    HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = 303,

    /** @brief UIDs of run-only intermediate tensors */
    HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = 304,

    /** @brief JSON representation of the execution plan */
    HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION = 305,

    /** @brief Kernel cache for this plan */
    HIPDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE = 306,

    /** @brief Device properties for this plan */
    HIPDNN_ATTR_EXECUTION_PLAN_DEVICEPROP = 307,

    /** @} */

    /**
     * @name Intermediate Info Attributes (400-499)
     * Attributes for intermediate tensor information
     * @{
     */

    /** @brief Unique ID of the intermediate tensor */
    HIPDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID = 400,

    /** @brief Size of the intermediate tensor in bytes */
    HIPDNN_ATTR_INTERMEDIATE_INFO_SIZE = 401,

    /** @brief UIDs of dependent input data */
    HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS = 402,

    /** @brief Dependent attribute identifiers */
    HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = 403,

    /** @} */

    /**
     * @name Knob Choice Attributes (500-599)
     * Attributes for HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR
     * @{
     */

    /** @brief Type of the knob */
    HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE = 500,

    /** @brief Selected value for the knob */
    HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE = 501,

    /** @} */

    /**
     * @name Operation Graph Attributes (600-699)
     * Attributes for HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
     * @{
     */

    /** @brief hipDNN handle for this graph */
    HIPDNN_ATTR_OPERATIONGRAPH_HANDLE = 600,

    /** @brief Array of operations in the graph */
    HIPDNN_ATTR_OPERATIONGRAPH_OPS = 601,

    /** @brief Total number of engines available globally */
    HIPDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = 602,

    /** @brief Whether dynamic shapes are enabled for this graph */
    HIPDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED = 603,

    /** @brief Compute data type for the operation graph (hipdnnDataType_t, extension) */
    HIPDNN_ATTR_OPERATIONGRAPH_COMPUTE_DATA_TYPE_EXT = 604,

    /** @brief Intermediate data type for the operation graph (hipdnnDataType_t, extension) */
    HIPDNN_ATTR_OPERATIONGRAPH_INTERMEDIATE_DATA_TYPE_EXT = 605,

    /** @brief I/O data type for the operation graph (hipdnnDataType_t, extension) */
    HIPDNN_ATTR_OPERATIONGRAPH_IO_DATA_TYPE_EXT = 606,

    /** @brief Preferred engine ID for execution plan selection (int64_t, extension) */
    HIPDNN_ATTR_OPERATIONGRAPH_PREFERRED_ENGINE_ID_EXT = 607,

    /** @} */

    /**
     * @name Variant Pack Attributes (700-799)
     * Attributes for HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR
     * @{
     */

    /** @brief Tensor unique IDs for data pointer mapping */
    HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = 700,

    /** @brief Device pointers to tensor data */
    HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 701,

    /** @brief Intermediate tensor data pointers */
    HIPDNN_ATTR_VARIANT_PACK_INTERMEDIATES = 702,

    /** @brief Workspace pointer for execution */
    HIPDNN_ATTR_VARIANT_PACK_WORKSPACE = 703,

    /** @} */

    /**
     * @name Layout Info Attributes (800-899)
     * Attributes for tensor layout information
     * @{
     */

    /** @brief Tensor UID for layout query */
    HIPDNN_ATTR_LAYOUT_INFO_TENSOR_UID = 800,

    /** @brief Supported layout types */
    HIPDNN_ATTR_LAYOUT_INFO_TYPES = 801,

    /** @} */

    /**
     * @name Knob Info Attributes (900-999)
     * Attributes for querying knob metadata
     * @{
     */

    /** @brief Type identifier of the knob */
    HIPDNN_ATTR_KNOB_INFO_TYPE = 900,

    /** @brief Maximum allowed value for the knob */
    HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE = 901,

    /** @brief Minimum allowed value for the knob */
    HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE = 902,

    /** @brief Step size for valid knob values */
    HIPDNN_ATTR_KNOB_INFO_STRIDE = 903,

    /** @} */

    /**
     * @name Engine Attributes (1000-1099)
     * Attributes for HIPDNN_BACKEND_ENGINE_DESCRIPTOR
     * @{
     */

    /** @brief Operation graph this engine supports */
    HIPDNN_ATTR_ENGINE_OPERATION_GRAPH = 1000,

    /** @brief Global index of this engine */
    HIPDNN_ATTR_ENGINE_GLOBAL_INDEX = 1001,

    /** @brief Available knob information for this engine */
    HIPDNN_ATTR_ENGINE_KNOB_INFO = 1002,

    /** @brief Numerical behavior notes (precision guarantees) */
    HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE = 1003,

    /** @brief Layout information for this engine */
    HIPDNN_ATTR_ENGINE_LAYOUT_INFO = 1004,

    /** @brief Behavioral notes for this engine */
    HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE = 1005,

    /** @brief Target SM count for this engine */
    HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET = 1006,

    /** @brief Device properties for this engine */
    HIPDNN_ATTR_ENGINE_DEVICEPROP = 1007,

    /** @} */

    /**
     * @name Kernel Cache Attributes (1100-1199)
     * Attributes for kernel caching functionality
     * @{
     */

    /** @brief Whether engine config kernel is cached */
    HIPDNN_ATTR_KERNEL_CACHE_IS_ENGINECFG_KERNEL_CACHED = 1100,

    /** @} */

    /**
     * @name Device Properties Attributes (1200-1299)
     * Attributes for HIPDNN_BACKEND_DEVICEPROP_DESCRIPTOR
     * @{
     */

    /** @brief HIP device ID */
    HIPDNN_ATTR_DEVICEPROP_DEVICE_ID = 1200,

    /** @brief hipDNN handle associated with device */
    HIPDNN_ATTR_DEVICEPROP_HANDLE = 1201,

    /** @brief JSON representation of device properties */
    HIPDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION = 1202,

    /** @} */

    /**
     * @name Tensor Attributes (1300-1399)
     * Attributes for HIPDNN_BACKEND_TENSOR_DESCRIPTOR
     * @{
     */

    /** @brief Unique ID for this tensor */
    HIPDNN_ATTR_TENSOR_UNIQUE_ID = 1300,

    /** @brief Tensor name (extension) */
    HIPDNN_ATTR_TENSOR_NAME_EXT = 1301,

    /** @brief Data type of tensor elements (hipdnnDataType_t) */
    HIPDNN_ATTR_TENSOR_DATA_TYPE = 1302,

    /** @brief Tensor dimensions */
    HIPDNN_ATTR_TENSOR_DIMENSIONS = 1303,

    /** @brief Tensor strides */
    HIPDNN_ATTR_TENSOR_STRIDES = 1304,

    /** @brief Whether this tensor is virtual */
    HIPDNN_ATTR_TENSOR_IS_VIRTUAL = 1305,

    /** @brief Pass-by-value tensor data (extension) */
    HIPDNN_ATTR_TENSOR_VALUE_EXT = 1306,

    /** @} */

    /**
     * @name Convolution Forward Operation Attributes (1400-1499)
     * Attributes for HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
     * @{
     */

    /** @brief Weight tensor for forward convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W = 1400,

    /** @brief Input tensor for forward convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X = 1401,

    /** @brief Output tensor for forward convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y = 1402,

    /** @} */

    /**
     * @name Shared Convolution Descriptor Attributes (1500-1599)
     * Attributes shared across convolution operation descriptors (forward,
     * dgrad, wgrad). These are set directly on the operation descriptor.
     * @{
     */

    /** @brief Compute data type for convolution */
    HIPDNN_ATTR_CONVOLUTION_COMP_TYPE = 1500,

    /** @brief Convolution mode (e.g., cross-correlation) */
    HIPDNN_ATTR_CONVOLUTION_CONV_MODE = 1501,

    /** @brief Dilation values for each spatial dimension */
    HIPDNN_ATTR_CONVOLUTION_DILATIONS = 1502,

    /** @brief Filter stride values for each spatial dimension */
    HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES = 1503,

    /** @brief Post-padding values for each spatial dimension */
    HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS = 1504,

    /** @brief Pre-padding values for each spatial dimension */
    HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS = 1505,

    /** @} */

    /**
     * @name Convolution Backward Filter Operation Attributes (1600-1699)
     * Attributes for HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR
     * @{
     */

    /** @brief Input tensor for backward filter convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_X = 1600,

    /** @brief Output gradient tensor for backward filter convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DY = 1601,

    /** @brief Weight gradient tensor for backward filter convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DW = 1602,

    /** @} */

    /**
     * @name Convolution Backward Operation Attributes (1700-1799)
     * Attributes for HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DESCRIPTOR
     * @{
     */

    /** @brief Output gradient tensor for backward data convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DY = 1700,

    /** @brief Weight tensor for backward data convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_W = 1701,

    /** @brief Input gradient tensor for backward data convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DX = 1702,

    /** @} */

    /**
     * @name Batchnorm Inference Operation Attributes (1800-1899)
     * Attributes for HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_EXT_DESCRIPTOR
     * @{
     */

    /** @brief Input tensor for batchnorm inference */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT = 1800,

    /** @brief Mean tensor for batchnorm inference */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT = 1801,

    /** @brief Inverse variance tensor for batchnorm inference */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT = 1802,

    /** @brief Scale tensor for batchnorm inference */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT = 1803,

    /** @brief Bias tensor for batchnorm inference */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT = 1804,

    /** @brief Output tensor for batchnorm inference */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT = 1805,

    /** @brief Compute data type for batchnorm inference */
    HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT = 1806,

    /** @} */

    /**
     * @name Extension Attributes (60000+)
     * hipDNN-specific extension attributes
     * @{
     */

    /**
     * @brief Serialized knob info as FlatBuffer (extension)
     *
     * Used to retrieve knob information in serialized FlatBuffer format.
     * Type: HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT
     */
    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT = 60000,

    /**
     * @brief Serialized knob choice as FlatBuffer (extension)
     *
     * Used to set knob values in serialized FlatBuffer format.
     * Type: HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT
     */
    HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT = 60100,

    /** @} */

} hipdnnBackendAttributeName_t;

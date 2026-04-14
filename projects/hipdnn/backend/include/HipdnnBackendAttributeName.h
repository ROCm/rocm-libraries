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
 * - 1300-1399: Tensor attributes
 * - 1400-1499: Convolution forward operation attributes
 * - 1500-1599: Shared convolution descriptor attributes
 * - 1600-1699: Convolution backward filter operation attributes
 * - 1700-1799: Convolution backward data operation attributes
 * - 1800-1899: Batchnorm inference operation attributes
 * - 1900-1999: Batchnorm inference variance ext operation attributes
 * - 2000-2099: Batchnorm backward ext operation attributes
 * - 2100-2199: Shared batchnorm backward ext attributes
 * - 2200-2299: Pointwise operation attributes
 * - 2300-2399: Shared pointwise descriptor attributes
 * - 2400-2499: RMSNorm operation attributes
 * - 2500-2599: Matmul operation attributes
 * - 2600-2699: SDPA forward propagation operation attributes
 * - 2700-2799: Layernorm operation attributes
 * - 2800-2899: Block scale quantize operation attributes
 * - 2807-2899: Block scale dequantize operation attributes
 * - 2900-2913: Batchnorm training forward operation attributes
 * - 3000-3099: Custom op operation attributes
 * - 3100-3199: SDPA backward propagation operation attributes
 * - 3200-3299: Reduction operation attributes
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

    /** @brief Find first mode: stop after finding any applicable engine (bool, extension) */
    HIPDNN_ATTR_ENGINEHEUR_FIND_FIRST_EXT = 105,

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

    /** @brief Name of the operation graph (HIPDNN_TYPE_CHAR, extension) */
    HIPDNN_ATTR_OPERATIONGRAPH_NAME_EXT = 608,

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

    /** @brief Human-readable description of the knob */
    HIPDNN_ATTR_KNOB_INFO_DESCRIPTION = 904,

    /** @brief Default value for the knob (INT64, DOUBLE, or CHAR) */
    HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE = 905,

    /** @brief Whether this knob is deprecated */
    HIPDNN_ATTR_KNOB_INFO_DEPRECATED = 906,

    /** @brief Explicit list of valid integer values (INT64 array) */
    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT = 907,

    /** @brief Explicit list of valid string values (CHAR, flat null-separated buffer) */
    HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING = 908,

    /** @brief Maximum string length for string knobs (INT32) */
    HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH = 909,

    /**
     * @brief Type discriminator for the default value attribute (read-only, INT64)
     *
     * Returns the hipdnnBackendAttributeType_t value that should be used when
     * calling getAttribute() for HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE.
     * Possible values: HIPDNN_TYPE_INT64, HIPDNN_TYPE_DOUBLE, HIPDNN_TYPE_CHAR.
     * This eliminates the need to probe multiple types to discover the default
     * value type after reading a KnobDescriptor.
     */
    HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_TYPE = 910,

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
    HIPDNN_ATTR_ENGINE_CU_COUNT_TARGET_EXT = 1006,

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

    /** @brief Read-only: whether a pass-by-value scalar is set on this tensor (extension) */
    HIPDNN_ATTR_TENSOR_IS_BY_VALUE = 1307,

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
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X = 1600,

    /** @brief Output gradient tensor for backward filter convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = 1601,

    /** @brief Weight gradient tensor for backward filter convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = 1602,

    /** @} */

    /**
     * @name Convolution Backward Data Operation Attributes (1700-1799)
     * Attributes for HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
     * @{
     */

    /** @brief Output gradient tensor for backward data convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = 1700,

    /** @brief Weight tensor for backward data convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = 1701,

    /** @brief Input gradient tensor for backward data convolution */
    HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = 1702,

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
     * @name Batchnorm Inference Variance Ext Operation Attributes (1900-1999)
     * Attributes for HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_VARIANCE_DESCRIPTOR_EXT
     * @{
     */

    /** @brief Input tensor for batchnorm inference variance ext */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_X_EXT = 1900,

    /** @brief Mean tensor for batchnorm inference variance ext */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_MEAN_EXT = 1901,

    /** @brief Variance tensor for batchnorm inference variance ext */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_VARIANCE_EXT = 1902,

    /** @brief Scale tensor for batchnorm inference variance ext */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_SCALE_EXT = 1903,

    /** @brief Bias tensor for batchnorm inference variance ext */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_BIAS_EXT = 1904,

    /** @brief Output tensor for batchnorm inference variance ext */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_Y_EXT = 1905,

    /** @brief Epsilon tensor for batchnorm inference variance ext */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EPSILON_EXT = 1906,

    /** @brief Compute data type for batchnorm inference variance ext */
    HIPDNN_ATTR_BATCHNORM_INF_VAR_COMP_TYPE_EXT = 1907,

    /** @} */

    /**
     * @name Batchnorm Backward Ext Operation Attributes (2000-2099)
     * Attributes for HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT
     * @{
     */

    /** @brief Gradient input tensor (dy) for batchnorm backward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT = 2000,

    /** @brief Input tensor (x) for batchnorm backward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT = 2001,

    /** @brief Scale tensor for batchnorm backward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT = 2002,

    /** @brief Gradient output tensor (dx) for batchnorm backward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT = 2003,

    /** @brief Scale gradient tensor (dscale) for batchnorm backward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT = 2004,

    /** @brief Bias gradient tensor (dbias) for batchnorm backward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT = 2005,

    /** @brief Saved mean tensor from forward pass */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT = 2006,

    /** @brief Saved inverse variance tensor from forward pass */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT = 2007,

    /** @} */

    /**
     * @name Shared Batchnorm Backward Ext Attributes (2100-2199)
     * @{
     */

    /** @brief Compute data type for batchnorm backward */
    HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT = 2100,

    /** @brief Peer statistics tensor array for multi-GPU batchnorm backward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT = 2101,

    /** @} */

    /**
     * @name Pointwise Operation Attributes (2200-2299)
     * Attributes for HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR
     * @{
     */

    /** @brief Primary input tensor for pointwise operation */
    HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT = 2200,

    /** @brief Output tensor for pointwise operation */
    HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT = 2201,

    /** @brief Secondary input tensor for pointwise operation (binary/ternary) */
    HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT = 2202,

    /** @brief Tertiary input tensor for pointwise operation (ternary) */
    HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT = 2203,

    /** @} */

    /**
     * @name Shared Pointwise Descriptor Attributes (2300-2399)
     * Attributes shared across pointwise operation descriptors.
     * These are set directly on the operation descriptor.
     * @{
     */

    /** @brief Pointwise operation mode */
    HIPDNN_ATTR_POINTWISE_MODE = 2300,

    /** @brief Lower clip value for ReLU activation */
    HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP = 2301,

    /** @brief Upper clip value for ReLU activation */
    HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP = 2302,

    /** @brief Lower clip slope for leaky ReLU activation */
    HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = 2303,

    /** @brief Beta parameter for Swish activation */
    HIPDNN_ATTR_POINTWISE_SWISH_BETA = 2304,

    /** @brief Alpha parameter for ELU activation */
    HIPDNN_ATTR_POINTWISE_ELU_ALPHA = 2305,

    /** @brief Beta parameter for Softplus activation */
    HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA = 2306,

    /** @brief Compute data type for pointwise operation */
    HIPDNN_ATTR_POINTWISE_MATH_PREC = 2307,

    /** @brief Axis index for pointwise operation */
    HIPDNN_ATTR_POINTWISE_AXIS = 2308,

    /** @} */

    /**
     * @name RMSNorm Operation Attributes (2400-2499)
     * Attributes for HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT
     * @{
     */

    /** @brief Input tensor (X) for rmsnorm */
    HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT = 2400,

    /** @brief Scale tensor for rmsnorm */
    HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT = 2401,

    /** @brief Epsilon tensor for rmsnorm */
    HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT = 2402,

    /** @brief Output tensor (Y) for rmsnorm */
    HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT = 2403,

    /** @brief Bias tensor for rmsnorm (optional) */
    HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT = 2404,

    /** @brief Inverse RMS tensor for rmsnorm (optional, training only) */
    HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT = 2405,

    /** @brief Forward phase for rmsnorm (TRAINING or INFERENCE) */
    HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT = 2406,

    /** @brief Compute data type for rmsnorm.
     *  Note: intentionally omits OPERATION_ prefix to match the BatchNorm
     *  inference convention (HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT). */
    HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT = 2407,

    /** @} */

    /**
     * @name Matmul Operation Attributes (2500-2599)
     * Attributes for HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR
     * @{
     */

    /** @brief Left input matrix tensor (A) for matmul */
    HIPDNN_ATTR_OPERATION_MATMUL_ADESC = 2500,

    /** @brief Right input matrix tensor (B) for matmul */
    HIPDNN_ATTR_OPERATION_MATMUL_BDESC = 2501,

    /** @brief Output matrix tensor (C) for matmul */
    HIPDNN_ATTR_OPERATION_MATMUL_CDESC = 2502,

    /** @brief Compute data type for matmul */
    HIPDNN_ATTR_MATMUL_COMP_TYPE = 2503,

    /** @} */

    /**
     * @name SDPA Forward Propagation Operation Attributes (2600-2699)
     * Attributes for HIPDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR
     * @{
     */

    /** @brief Q (query) tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_QDESC = 2600,

    /** @brief K (key) tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_KDESC = 2601,

    /** @brief V (value) tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_VDESC = 2602,

    /** @brief O (output) tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_ODESC = 2603,

    /** @brief Attention mask tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_ATTN_MASK_EXT = 2604,

    /** @brief Scale tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC = 2605,

    /** @brief Sequence length Q tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC = 2606,

    /** @brief Sequence length KV tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC = 2607,

    /** @brief Seed tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_SEED_EXT = 2608,

    /** @brief Offset tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_OFFSET_EXT = 2609,

    /** @brief Dropout mask tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_MASK_EXT = 2610,

    /** @brief Dropout scale tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SCALE_EXT = 2611,

    /** @brief Page table K tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC = 2612,

    /** @brief Page table V tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC = 2613,

    /** @brief Block mask tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC = 2614,

    /** @brief Sink token tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_SINK_TOKEN_EXT = 2615,

    /** @brief Descale Q tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_Q_EXT = 2616,

    /** @brief Descale K tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_K_EXT = 2617,

    /** @brief Descale V tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_V_EXT = 2618,

    /** @brief Descale S tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_S_EXT = 2619,

    /** @brief Scale S tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALE_S_EXT = 2620,

    /** @brief Scale O tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALE_O_EXT = 2621,

    /** @brief Stats output tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC = 2622,

    /** @brief Max output tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_MAX_EXT = 2623,

    /** @brief Sum exp output tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_SUM_EXP_EXT = 2624,

    /** @brief RNG dump output tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_RNG_DUMP_EXT = 2625,

    /** @brief Amax S output tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_AMAX_S_EXT = 2626,

    /** @brief Amax O output tensor for SDPA forward */
    HIPDNN_ATTR_OPERATION_SDPA_FWD_AMAX_O_EXT = 2627,

    /** @brief Whether to generate statistics (bool) */
    HIPDNN_ATTR_SDPA_FWD_GENERATE_STATS_EXT = 2628,

    /** @brief Whether to use ALiBi mask (bool) */
    HIPDNN_ATTR_SDPA_FWD_ALIBI_MASK_EXT = 2629,

    /** @brief Whether to use padding mask (bool) */
    HIPDNN_ATTR_SDPA_FWD_PADDING_MASK_EXT = 2630,

    /** @brief Whether to use causal mask (bool, deprecated) */
    HIPDNN_ATTR_SDPA_FWD_CAUSAL_MASK_EXT = 2631,

    /** @brief Whether to use causal mask bottom-right (bool, deprecated) */
    HIPDNN_ATTR_SDPA_FWD_CAUSAL_MASK_BOTTOM_RIGHT_EXT = 2632,

    /** @brief Dropout probability (float) */
    HIPDNN_ATTR_SDPA_FWD_DROPOUT_PROBABILITY_EXT = 2633,

    /** @brief Attention scale value (float) */
    HIPDNN_ATTR_SDPA_FWD_ATTN_SCALE_VALUE_EXT = 2634,

    /** @brief Left bound for sliding window (int64) */
    HIPDNN_ATTR_SDPA_FWD_LEFT_BOUND_EXT = 2635,

    /** @brief Right bound for sliding window (int64) */
    HIPDNN_ATTR_SDPA_FWD_RIGHT_BOUND_EXT = 2636,

    /** @brief Maximum sequence length KV (int32_t) */
    HIPDNN_ATTR_SDPA_FWD_MAX_SEQ_LEN_KV_EXT = 2637,

    /** @brief Diagonal alignment mode (hipdnnDiagonalAlignment_t) */
    HIPDNN_ATTR_SDPA_FWD_DIAGONAL_ALIGNMENT_EXT = 2638,

    /** @brief MMA core mode (hipdnnDataType_t) */
    HIPDNN_ATTR_SDPA_FWD_MMA_CORE_MODE_EXT = 2639,

    /** @brief Attention implementation mode (hipdnnAttentionImplementation_t) */
    HIPDNN_ATTR_SDPA_FWD_IMPLEMENTATION_EXT = 2640,

    /** @brief Compute data type for SDPA forward */
    HIPDNN_ATTR_SDPA_FWD_COMP_TYPE_EXT = 2641,

    /** @} */

    /**
     * @name Layernorm Operation Attributes (2700-2799)
     * Attributes for HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT
     * @{
     */

    /** @brief Input tensor for layernorm */
    HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT = 2700,

    /** @brief Scale tensor for layernorm */
    HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT = 2701,

    /** @brief Bias tensor for layernorm */
    HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT = 2702,

    /** @brief Epsilon tensor for layernorm */
    HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT = 2703,

    /** @brief Output tensor for layernorm */
    HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT = 2704,

    /** @brief Mean output tensor for layernorm (optional) */
    HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT = 2705,

    /** @brief Inverse variance output tensor for layernorm (optional) */
    HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT = 2706,

    /** @brief Forward phase for layernorm (TRAINING or INFERENCE) */
    HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT = 2707,

    /** @brief Math precision (compute data type) for layernorm */
    HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT = 2708,

    /** @brief Number of normalized dimensions for layernorm */
    HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT = 2709,
    /** @} */

    /**
     * @name Block Scale Quantize Operation Attributes (2800-2899)
     * Attributes for HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR
     * @{
     */

    /** @brief Input tensor (X) for block scale quantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC = 2800,

    /** @brief Output tensor (Y) for block scale quantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC = 2801,

    /** @brief Scale tensor for block scale quantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC = 2802,

    /** @brief Block size parameter for block scale quantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE = 2803,

    /** @brief Axis parameter for block scale quantize (optional) */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT = 2804,

    /** @brief Transpose parameter for block scale quantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT = 2805,

    /** @brief Math precision for block scale quantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC = 2806,

    /** @} */

    /**
     * @name BlockScaleDequantize Operation Attributes (2807-2899)
     * Attributes for HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR
     * @{
     */

    /** @brief Input tensor for block scale dequantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC = 2807,

    /** @brief Scale tensor for block scale dequantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC = 2808,

    /** @brief Output tensor for block scale dequantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC = 2809,

    /** @brief Block sizes for each dimension */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE = 2810,

    /** @brief Whether scale is negative */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE = 2811,

    /** @brief Math precision for block scale dequantize */
    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC = 2812,

    /** @} */

    /**
     * @name Batchnorm Training Forward Operation Attributes (2900-2913)
     * Attributes for HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT
     * @{
     */

    /** @brief Input tensor (X) for batchnorm training forward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT = 2900,

    /** @brief Scale tensor for batchnorm training forward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT = 2901,

    /** @brief Bias tensor for batchnorm training forward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT = 2902,

    /** @brief Epsilon tensor for batchnorm training forward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT = 2903,

    /** @brief Output tensor (Y) for batchnorm training forward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT = 2904,

    /** @brief Previous running mean tensor (optional) */
    HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT = 2905,

    /** @brief Previous running variance tensor (optional) */
    HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT = 2906,

    /** @brief Momentum tensor (optional, required if running stats provided) */
    HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT = 2907,

    /** @brief Batch mean output tensor (optional) */
    HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT = 2908,

    /** @brief Batch inverse variance output tensor (optional) */
    HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT = 2909,

    /** @brief Next running mean output tensor (optional) */
    HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT = 2910,

    /** @brief Next running variance output tensor (optional) */
    HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT = 2911,

    /** @brief Math precision (compute data type) for batchnorm training forward */
    HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT = 2912,

    /** @brief Peer statistics tensor array for multi-GPU batchnorm training forward */
    HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT = 2913,

    /** @} */

    /**
     * @name Custom Op Operation Attributes (3000-3099)
     * Attributes for HIPDNN_BACKEND_OPERATION_CUSTOM_OP_DESCRIPTOR_EXT
     * @{
     */

    /** @brief Input tensor array for custom op */
    HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT = 3000,

    /** @brief Output tensor array for custom op */
    HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT = 3001,

    /** @brief Plugin identifier string for custom op (e.g. "example.rope") */
    HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT = 3002,

    /** @brief Opaque byte payload for custom op */
    HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT = 3003,

    /** @brief Compute data type for custom op */
    HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT = 3004,

    /** @} */

    /**
     * @name SDPA Backward Propagation Operation Attributes (3100-3199)
     * Attributes for HIPDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR_EXT
     * @{
     */

    /** @brief Q (query) tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_Q_EXT = 3100,

    /** @brief K (key) tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_K_EXT = 3101,

    /** @brief V (value) tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_V_EXT = 3102,

    /** @brief O (output) tensor from forward pass for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_O_EXT = 3103,

    /** @brief dO (output gradient) tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_DO_EXT = 3104,

    /** @brief Stats tensor from forward pass for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_STATS_EXT = 3105,

    /** @brief dQ (query gradient) output tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_DQ_EXT = 3106,

    /** @brief dK (key gradient) output tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_DK_EXT = 3107,

    /** @brief dV (value gradient) output tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_DV_EXT = 3108,

    /** @brief Scale tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_SCALE_EXT = 3109,

    /** @brief Attention mask tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_ATTN_MASK_EXT = 3110,

    /** @brief Sequence length Q tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_Q_EXT = 3111,

    /** @brief Sequence length KV tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KV_EXT = 3112,

    /** @brief Seed tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_SEED_EXT = 3113,

    /** @brief Offset tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_OFFSET_EXT = 3114,

    /** @brief Dropout mask tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_MASK_EXT = 3115,

    /** @brief Dropout scale tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_EXT = 3116,

    /** @brief Dropout scale inverse tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_INV_EXT = 3117,

    /** @brief dBias (gradient of attention bias) output tensor for SDPA backward */
    HIPDNN_ATTR_OPERATION_SDPA_BWD_DBIAS_EXT = 3118,

    /** @brief Whether to use ALiBi mask (bool) */
    HIPDNN_ATTR_SDPA_BWD_ALIBI_MASK_EXT = 3119,

    /** @brief Whether to use padding mask (bool) */
    HIPDNN_ATTR_SDPA_BWD_PADDING_MASK_EXT = 3120,

    /** @brief Whether to use causal mask (bool, deprecated) */
    HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_EXT = 3121,

    /** @brief Whether to use causal mask bottom-right (bool, deprecated) */
    HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_BOTTOM_RIGHT_EXT = 3122,

    /** @brief Dropout probability (float) */
    HIPDNN_ATTR_SDPA_BWD_DROPOUT_PROBABILITY_EXT = 3123,

    /** @brief Attention scale value (float) */
    HIPDNN_ATTR_SDPA_BWD_ATTN_SCALE_VALUE_EXT = 3124,

    /** @brief Left bound for sliding window (int64) */
    HIPDNN_ATTR_SDPA_BWD_LEFT_BOUND_EXT = 3125,

    /** @brief Right bound for sliding window (int64) */
    HIPDNN_ATTR_SDPA_BWD_RIGHT_BOUND_EXT = 3126,

    /** @brief Diagonal alignment mode (hipdnnDiagonalAlignment_t) */
    HIPDNN_ATTR_SDPA_BWD_DIAGONAL_ALIGNMENT_EXT = 3127,

    /** @brief Compute data type for SDPA backward */
    HIPDNN_ATTR_SDPA_BWD_COMP_TYPE_EXT = 3128,

    /** @} */

    /**
     * @name Reduction Operation Attributes (3200-3299)
     * Attributes for HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR
     * @{
     */

    /** @brief Input tensor (X) for reduction */
    HIPDNN_ATTR_OPERATION_REDUCTION_XDESC = 3200,

    /** @brief Output tensor (Y) for reduction */
    HIPDNN_ATTR_OPERATION_REDUCTION_YDESC = 3201,

    /** @brief Reduction operator (hipdnnReduceTensorOp_t) */
    HIPDNN_ATTR_REDUCTION_OPERATOR = 3202,

    /** @brief Compute data type for reduction */
    HIPDNN_ATTR_REDUCTION_COMP_TYPE = 3203,

    /** @brief Whether reduction is deterministic (bool) */
    HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC = 3204,

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
    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE = 60000,

    /**
     * @brief Serialized knob choice as FlatBuffer (extension)
     *
     * Used to set knob values in serialized FlatBuffer format.
     * Type: HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT
     */
    HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE = 60100,

    /**
     * @brief Operation type of an operation descriptor (read-only extension)
     *
     * Returns the hipdnnOperationType_ext_t of an operation descriptor, enabling
     * type-based dispatch without trial-and-error probing.
     * Type: HIPDNN_TYPE_OPERATION_TYPE_EXT
     */
    HIPDNN_ATTR_OPERATION_TYPE_EXT = 60200,

    /**
     * @brief Name of an operation descriptor (extension)
     *
     * Gets or sets a human-readable name for an operation node, useful for
     * debugging, logging, and round-tripping through serialized graphs.
     * Type: HIPDNN_TYPE_CHAR
     */
    HIPDNN_ATTR_OPERATION_NAME_EXT = 60300,

    /** @} */

} hipdnnBackendAttributeName_t;

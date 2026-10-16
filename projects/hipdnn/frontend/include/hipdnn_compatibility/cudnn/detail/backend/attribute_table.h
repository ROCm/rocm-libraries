// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

enum class CountRule : std::uint8_t
{
    SCALAR, // exactly one element
    ARRAY, // 0..maxCount elements; 0 is stored as empty and judged at finalize
    CHAR_STRING, // 0..maxCount chars; a trailing NUL is accepted and dropped
};

// Constraint on the descriptors stored in a BACKEND_DESCRIPTOR attribute.
enum class ReferentKind : std::uint8_t
{
    NONE,
    EXACT,
    ANY_OPERATION,
    // cuDNN descriptor type the shim does not declare (device properties, kernel
    // cache): SET reports NOT_SUPPORTED.
    UNAVAILABLE,
};

// Bound for arrays upstream sizes at runtime; keeps count * element size far
// from overflow.
inline constexpr std::int64_t UNBOUNDED_COUNT = std::numeric_limits<std::int32_t>::max();

// Typing rule for one attribute, taken from the upstream cudnn-frontend v1.24.0
// call sites (tests/data/cudnn_v8/cudnn_v8_attribute_usage.inc).
struct AttributeInfo
{
    const char* name;
    cudnnBackendDescriptorType_t owner;
    std::array<cudnnBackendAttributeType_t, 2> types;
    std::uint8_t typeCount;
    CountRule countRule;
    std::int64_t maxCount;
    ReferentKind referentKind;
    cudnnBackendDescriptorType_t referent;
    // False for values the descriptor computes at finalize (GET-only).
    bool settable;
    bool hasDefault;
    // Read as the first accepted type (BOOLEAN, INT64 or a C enum).
    std::int64_t defaultValue;

    static constexpr AttributeInfo row(const char* attributeName,
                                       cudnnBackendDescriptorType_t ownerType,
                                       cudnnBackendAttributeType_t type)
    {
        return AttributeInfo{attributeName,
                             ownerType,
                             {type, type},
                             1,
                             CountRule::SCALAR,
                             1,
                             ReferentKind::NONE,
                             ownerType,
                             true,
                             false,
                             0};
    }

    constexpr AttributeInfo orType(cudnnBackendAttributeType_t type) const
    {
        AttributeInfo result = *this;
        result.types[1] = type;
        result.typeCount = 2;
        return result;
    }

    constexpr AttributeInfo asArray(std::int64_t max) const
    {
        AttributeInfo result = *this;
        result.countRule = CountRule::ARRAY;
        result.maxCount = max;
        return result;
    }

    constexpr AttributeInfo asString(std::int64_t max) const
    {
        AttributeInfo result = *this;
        result.countRule = CountRule::CHAR_STRING;
        result.maxCount = max;
        return result;
    }

    constexpr AttributeInfo refersTo(cudnnBackendDescriptorType_t type) const
    {
        AttributeInfo result = *this;
        result.referentKind = ReferentKind::EXACT;
        result.referent = type;
        return result;
    }

    constexpr AttributeInfo refersToAnyOperation() const
    {
        AttributeInfo result = *this;
        result.referentKind = ReferentKind::ANY_OPERATION;
        return result;
    }

    constexpr AttributeInfo refersToUnavailable() const
    {
        AttributeInfo result = *this;
        result.referentKind = ReferentKind::UNAVAILABLE;
        return result;
    }

    constexpr AttributeInfo withDefault(std::int64_t value) const
    {
        AttributeInfo result = *this;
        result.hasDefault = true;
        result.defaultValue = value;
        return result;
    }

    constexpr AttributeInfo computed() const
    {
        AttributeInfo result = *this;
        result.settable = false;
        return result;
    }

    constexpr bool accepts(cudnnBackendAttributeType_t type) const
    {
        return types[0] == type || (typeCount == 2 && types[1] == type);
    }
};

inline const char* descriptorTypeName(cudnnBackendDescriptorType_t type)
{
    switch(type)
    {
    case CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
        return "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR";
    case CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
        return "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR";
    case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
        return "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR";
    case CUDNN_BACKEND_ENGINE_DESCRIPTOR:
        return "CUDNN_BACKEND_ENGINE_DESCRIPTOR";
    case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
        return "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR";
    case CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
        return "CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR";
    case CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
        return "CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR";
    case CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
        return "CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR";
    case CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR:
        return "CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR";
    case CUDNN_BACKEND_MATMUL_DESCRIPTOR:
        return "CUDNN_BACKEND_MATMUL_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR";
    case CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR:
        return "CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR";
    case CUDNN_BACKEND_POINTWISE_DESCRIPTOR:
        return "CUDNN_BACKEND_POINTWISE_DESCRIPTOR";
    case CUDNN_BACKEND_REDUCTION_DESCRIPTOR:
        return "CUDNN_BACKEND_REDUCTION_DESCRIPTOR";
    case CUDNN_BACKEND_RESAMPLE_DESCRIPTOR:
        return "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR";
    case CUDNN_BACKEND_RNG_DESCRIPTOR:
        return "CUDNN_BACKEND_RNG_DESCRIPTOR";
    case CUDNN_BACKEND_TENSOR_DESCRIPTOR:
        return "CUDNN_BACKEND_TENSOR_DESCRIPTOR";
    case CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
        return "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR";
    default:
        return "unknown cudnnBackendDescriptorType_t";
    }
}

inline const char* attributeTypeName(cudnnBackendAttributeType_t type)
{
    switch(type)
    {
    case CUDNN_TYPE_BACKEND_DESCRIPTOR:
        return "CUDNN_TYPE_BACKEND_DESCRIPTOR";
    case CUDNN_TYPE_BEHAVIOR_NOTE:
        return "CUDNN_TYPE_BEHAVIOR_NOTE";
    case CUDNN_TYPE_BN_FINALIZE_STATS_MODE:
        return "CUDNN_TYPE_BN_FINALIZE_STATS_MODE";
    case CUDNN_TYPE_BOOLEAN:
        return "CUDNN_TYPE_BOOLEAN";
    case CUDNN_TYPE_CHAR:
        return "CUDNN_TYPE_CHAR";
    case CUDNN_TYPE_CONVOLUTION_MODE:
        return "CUDNN_TYPE_CONVOLUTION_MODE";
    case CUDNN_TYPE_DATA_TYPE:
        return "CUDNN_TYPE_DATA_TYPE";
    case CUDNN_TYPE_DOUBLE:
        return "CUDNN_TYPE_DOUBLE";
    case CUDNN_TYPE_FLOAT:
        return "CUDNN_TYPE_FLOAT";
    case CUDNN_TYPE_FRACTION:
        return "CUDNN_TYPE_FRACTION";
    case CUDNN_TYPE_GENSTATS_MODE:
        return "CUDNN_TYPE_GENSTATS_MODE";
    case CUDNN_TYPE_HANDLE:
        return "CUDNN_TYPE_HANDLE";
    case CUDNN_TYPE_HEUR_MODE:
        return "CUDNN_TYPE_HEUR_MODE";
    case CUDNN_TYPE_INT32:
        return "CUDNN_TYPE_INT32";
    case CUDNN_TYPE_INT64:
        return "CUDNN_TYPE_INT64";
    case CUDNN_TYPE_KNOB_TYPE:
        return "CUDNN_TYPE_KNOB_TYPE";
    case CUDNN_TYPE_NAN_PROPOGATION:
        return "CUDNN_TYPE_NAN_PROPOGATION";
    case CUDNN_TYPE_NORM_FWD_PHASE:
        return "CUDNN_TYPE_NORM_FWD_PHASE";
    case CUDNN_TYPE_NORM_MODE:
        return "CUDNN_TYPE_NORM_MODE";
    case CUDNN_TYPE_NUMERICAL_NOTE:
        return "CUDNN_TYPE_NUMERICAL_NOTE";
    case CUDNN_TYPE_PADDING_MODE:
        return "CUDNN_TYPE_PADDING_MODE";
    case CUDNN_TYPE_POINTWISE_MODE:
        return "CUDNN_TYPE_POINTWISE_MODE";
    case CUDNN_TYPE_REDUCTION_OPERATOR_TYPE:
        return "CUDNN_TYPE_REDUCTION_OPERATOR_TYPE";
    case CUDNN_TYPE_RESAMPLE_MODE:
        return "CUDNN_TYPE_RESAMPLE_MODE";
    case CUDNN_TYPE_RESHAPE_MODE:
        return "CUDNN_TYPE_RESHAPE_MODE";
    case CUDNN_TYPE_RNG_DISTRIBUTION:
        return "CUDNN_TYPE_RNG_DISTRIBUTION";
    case CUDNN_TYPE_TENSOR_REORDERING_MODE:
        return "CUDNN_TYPE_TENSOR_REORDERING_MODE";
    case CUDNN_TYPE_VOID_PTR:
        return "CUDNN_TYPE_VOID_PTR";
    default:
        return "unknown cudnnBackendAttributeType_t";
    }
}

inline bool isOperationDescriptorType(cudnnBackendDescriptorType_t type)
{
    switch(type)
    {
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR:
        return true;
    default:
        return false;
    }
}

// Number of declared enumerators, excluding *_COUNT sentinels, for attribute
// types whose element is a 4-byte C enum; valid values are 0..count-1.
inline std::optional<std::int32_t> enumValueCount(cudnnBackendAttributeType_t type)
{
    switch(type)
    {
    case CUDNN_TYPE_BEHAVIOR_NOTE:
        return CUDNN_BEHAVIOR_NOTE_TYPE_COUNT;
    case CUDNN_TYPE_BN_FINALIZE_STATS_MODE:
        return CUDNN_BN_FINALIZE_STATISTICS_TRAINING + 1;
    case CUDNN_TYPE_CONVOLUTION_MODE:
        return CUDNN_CROSS_CORRELATION + 1;
    case CUDNN_TYPE_DATA_TYPE:
        return CUDNN_DATA_UINT8x4 + 1;
    case CUDNN_TYPE_GENSTATS_MODE:
        return CUDNN_GENSTATS_SUM_SQSUM + 1;
    case CUDNN_TYPE_HEUR_MODE:
        return CUDNN_HEUR_MODES_COUNT;
    case CUDNN_TYPE_KNOB_TYPE:
        return CUDNN_KNOB_TYPE_COUNTS;
    case CUDNN_TYPE_NAN_PROPOGATION:
        return CUDNN_PROPAGATE_NAN + 1;
    case CUDNN_TYPE_NORM_FWD_PHASE:
        return CUDNN_NORM_FWD_TRAINING + 1;
    case CUDNN_TYPE_NORM_MODE:
        return CUDNN_RMS_NORM + 1;
    case CUDNN_TYPE_NUMERICAL_NOTE:
        return CUDNN_NUMERICAL_NOTE_TYPE_COUNT;
    case CUDNN_TYPE_PADDING_MODE:
        return CUDNN_ZERO_PAD + 1;
    case CUDNN_TYPE_POINTWISE_MODE:
        return CUDNN_POINTWISE_TANH_FWD + 1;
    case CUDNN_TYPE_REDUCTION_OPERATOR_TYPE:
        return CUDNN_REDUCE_TENSOR_NORM2 + 1;
    case CUDNN_TYPE_RESAMPLE_MODE:
        return CUDNN_RESAMPLE_NEAREST + 1;
    case CUDNN_TYPE_RESHAPE_MODE:
        return CUDNN_RESHAPE_VIEW_ONLY + 1;
    case CUDNN_TYPE_RNG_DISTRIBUTION:
        return CUDNN_RNG_DISTRIBUTION_UNIFORM + 1;
    case CUDNN_TYPE_TENSOR_REORDERING_MODE:
        return CUDNN_TENSOR_REORDERING_F8_128x4 + 1;
    default:
        return std::nullopt;
    }
}

inline bool isEnumAttributeType(cudnnBackendAttributeType_t type)
{
    return enumValueCount(type).has_value();
}

// One row per CUDNN_ATTR_* in the identifier fixture. Every element type,
// count bound and referent comes from an upstream call site; alpha/beta accept
// FLOAT or DOUBLE because upstream picks the type from the compute type.
inline std::optional<AttributeInfo> attributeInfo(cudnnBackendAttributeName_t name)
{
    switch(name)
    {
    case CUDNN_ATTR_CONVOLUTION_COMP_TYPE:
        return AttributeInfo::row("CUDNN_ATTR_CONVOLUTION_COMP_TYPE",
                                  CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                                  CUDNN_TYPE_DATA_TYPE);
    case CUDNN_ATTR_CONVOLUTION_CONV_MODE:
        return AttributeInfo::row("CUDNN_ATTR_CONVOLUTION_CONV_MODE",
                                  CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                                  CUDNN_TYPE_CONVOLUTION_MODE);
    case CUDNN_ATTR_CONVOLUTION_DILATIONS:
        return AttributeInfo::row("CUDNN_ATTR_CONVOLUTION_DILATIONS",
                                  CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .asArray(CUDNN_DIM_MAX);
    case CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        return AttributeInfo::row("CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES",
                                  CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .asArray(CUDNN_DIM_MAX);
    case CUDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        return AttributeInfo::row("CUDNN_ATTR_CONVOLUTION_POST_PADDINGS",
                                  CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .asArray(CUDNN_DIM_MAX);
    case CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        return AttributeInfo::row("CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS",
                                  CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .asArray(CUDNN_DIM_MAX);
    case CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS:
        return AttributeInfo::row("CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS",
                                  CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
                                  CUDNN_TYPE_INT64);
    case CUDNN_ATTR_ENGINECFG_ENGINE:
        return AttributeInfo::row("CUDNN_ATTR_ENGINECFG_ENGINE",
                                  CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    case CUDNN_ATTR_ENGINECFG_KNOB_CHOICES:
        return AttributeInfo::row("CUDNN_ATTR_ENGINECFG_KNOB_CHOICES",
                                  CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .asArray(CUDNN_KNOB_TYPE_COUNTS)
            .refersTo(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
    case CUDNN_ATTR_ENGINEHEUR_DEVICEPROP:
        return AttributeInfo::row("CUDNN_ATTR_ENGINEHEUR_DEVICEPROP",
                                  CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersToUnavailable();
    case CUDNN_ATTR_ENGINEHEUR_MODE:
        return AttributeInfo::row("CUDNN_ATTR_ENGINEHEUR_MODE",
                                  CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                                  CUDNN_TYPE_HEUR_MODE);
    case CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        return AttributeInfo::row("CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH",
                                  CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    case CUDNN_ATTR_ENGINEHEUR_RESULTS:
        return AttributeInfo::row("CUDNN_ATTR_ENGINEHEUR_RESULTS",
                                  CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .asArray(UNBOUNDED_COUNT)
            .refersTo(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR)
            .computed();
    case CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET:
        return AttributeInfo::row("CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET",
                                  CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                                  CUDNN_TYPE_INT32);
    case CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
        return AttributeInfo::row("CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE",
                                  CUDNN_BACKEND_ENGINE_DESCRIPTOR,
                                  CUDNN_TYPE_BEHAVIOR_NOTE)
            .asArray(CUDNN_BEHAVIOR_NOTE_TYPE_COUNT)
            .computed();
    case CUDNN_ATTR_ENGINE_GLOBAL_INDEX:
        return AttributeInfo::row(
            "CUDNN_ATTR_ENGINE_GLOBAL_INDEX", CUDNN_BACKEND_ENGINE_DESCRIPTOR, CUDNN_TYPE_INT64);
    case CUDNN_ATTR_ENGINE_KNOB_INFO:
        return AttributeInfo::row("CUDNN_ATTR_ENGINE_KNOB_INFO",
                                  CUDNN_BACKEND_ENGINE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .asArray(CUDNN_KNOB_TYPE_COUNTS)
            .refersTo(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR)
            .computed();
    case CUDNN_ATTR_ENGINE_NUMERICAL_NOTE:
        return AttributeInfo::row("CUDNN_ATTR_ENGINE_NUMERICAL_NOTE",
                                  CUDNN_BACKEND_ENGINE_DESCRIPTOR,
                                  CUDNN_TYPE_NUMERICAL_NOTE)
            .asArray(CUDNN_NUMERICAL_NOTE_TYPE_COUNT)
            .computed();
    case CUDNN_ATTR_ENGINE_OPERATION_GRAPH:
        return AttributeInfo::row("CUDNN_ATTR_ENGINE_OPERATION_GRAPH",
                                  CUDNN_BACKEND_ENGINE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    case CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        return AttributeInfo::row("CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG",
                                  CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    case CUDNN_ATTR_EXECUTION_PLAN_HANDLE:
        return AttributeInfo::row("CUDNN_ATTR_EXECUTION_PLAN_HANDLE",
                                  CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                                  CUDNN_TYPE_HANDLE);
    case CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
        return AttributeInfo::row("CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION",
                                  CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                                  CUDNN_TYPE_CHAR)
            .asString(UNBOUNDED_COUNT);
    case CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE:
        return AttributeInfo::row("CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE",
                                  CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersToUnavailable();
    case CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
        return AttributeInfo::row("CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE",
                                  CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .computed();
    case CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE:
        return AttributeInfo::row("CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE",
                                  CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR,
                                  CUDNN_TYPE_KNOB_TYPE);
    case CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE:
        return AttributeInfo::row("CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE",
                                  CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR,
                                  CUDNN_TYPE_INT64);
    case CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
        return AttributeInfo::row("CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE",
                                  CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .computed();
    case CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
        return AttributeInfo::row("CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE",
                                  CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .computed();
    case CUDNN_ATTR_KNOB_INFO_STRIDE:
        return AttributeInfo::row("CUDNN_ATTR_KNOB_INFO_STRIDE",
                                  CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .computed();
    case CUDNN_ATTR_KNOB_INFO_TYPE:
        return AttributeInfo::row("CUDNN_ATTR_KNOB_INFO_TYPE",
                                  CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
                                  CUDNN_TYPE_KNOB_TYPE)
            .computed();
    case CUDNN_ATTR_MATMUL_COMP_TYPE:
        return AttributeInfo::row(
            "CUDNN_ATTR_MATMUL_COMP_TYPE", CUDNN_BACKEND_MATMUL_DESCRIPTOR, CUDNN_TYPE_DATA_TYPE);
    case CUDNN_ATTR_MATMUL_PADDING_VALUE:
        return AttributeInfo::row(
            "CUDNN_ATTR_MATMUL_PADDING_VALUE", CUDNN_BACKEND_MATMUL_DESCRIPTOR, CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT:
        return AttributeInfo::row("CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT",
                                  CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .computed();
    case CUDNN_ATTR_OPERATIONGRAPH_HANDLE:
        return AttributeInfo::row("CUDNN_ATTR_OPERATIONGRAPH_HANDLE",
                                  CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                                  CUDNN_TYPE_HANDLE);
    case CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED:
        return AttributeInfo::row("CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED",
                                  CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                                  CUDNN_TYPE_BOOLEAN);
    case CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED:
        return AttributeInfo::row("CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED",
                                  CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                                  CUDNN_TYPE_BOOLEAN);
    case CUDNN_ATTR_OPERATIONGRAPH_OPS:
        return AttributeInfo::row("CUDNN_ATTR_OPERATIONGRAPH_OPS",
                                  CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .asArray(UNBOUNDED_COUNT)
            .refersToAnyOperation();
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_DATA_TYPE);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_DATA_TYPE);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BN_FINALIZE_STATS_MODE);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC",
                                  CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",
                                  CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC",
                                  CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
                                  CUDNN_TYPE_DATA_TYPE);
    case CUDNN_ATTR_OPERATION_GENSTATS_MODE:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_GENSTATS_MODE",
                                  CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
                                  CUDNN_TYPE_GENSTATS_MODE);
    case CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC",
                                  CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC",
                                  CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_GENSTATS_XDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_GENSTATS_XDESC",
                                  CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_MATMUL_ADESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_MATMUL_ADESC",
                                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_MATMUL_BDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_MATMUL_BDESC",
                                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_MATMUL_CDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_MATMUL_CDESC",
                                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_MATMUL_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_MATMUL_DESC",
                                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC",
                                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC",
                                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC",
                                  CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_MODE:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_MODE",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_NORM_MODE);
    case CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .asArray(UNBOUNDED_COUNT)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_BWD_XDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_BWD_XDESC",
                                  CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_MODE:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_MODE",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_NORM_MODE);
    case CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .asArray(UNBOUNDED_COUNT)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_PHASE:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_PHASE",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_NORM_FWD_PHASE);
    case CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_XDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_XDESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_NORM_FWD_YDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_NORM_FWD_YDESC",
                                  CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC",
                                  CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC",
                                  CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC",
                                  CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC",
                                  CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1",
                                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2",
                                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_POINTWISE_BDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_POINTWISE_BDESC",
                                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_POINTWISE_DXDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_POINTWISE_DXDESC",
                                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_POINTWISE_DYDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_POINTWISE_DYDESC",
                                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR",
                                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_POINTWISE_TDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_POINTWISE_TDESC",
                                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_POINTWISE_XDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_POINTWISE_XDESC",
                                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_POINTWISE_YDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_POINTWISE_YDESC",
                                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_REDUCTION_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_REDUCTION_DESC",
                                  CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_REDUCTION_XDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_REDUCTION_XDESC",
                                  CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_REDUCTION_YDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_REDUCTION_YDESC",
                                  CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
                                  CUDNN_TYPE_FLOAT)
            .orType(CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC",
                                  CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESHAPE_MODE:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESHAPE_MODE",
                                  CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR,
                                  CUDNN_TYPE_RESHAPE_MODE);
    case CUDNN_ATTR_OPERATION_RESHAPE_XDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESHAPE_XDESC",
                                  CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RESHAPE_YDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RESHAPE_YDESC",
                                  CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RNG_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RNG_DESC",
                                  CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_RNG_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC",
                                  CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RNG_SEED:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RNG_SEED",
                                  CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .orType(CUDNN_TYPE_INT64)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_OPERATION_RNG_YDESC:
        return AttributeInfo::row("CUDNN_ATTR_OPERATION_RNG_YDESC",
                                  CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_POINTWISE_AXIS:
        return AttributeInfo::row(
            "CUDNN_ATTR_POINTWISE_AXIS", CUDNN_BACKEND_POINTWISE_DESCRIPTOR, CUDNN_TYPE_INT64);
    case CUDNN_ATTR_POINTWISE_ELU_ALPHA:
        return AttributeInfo::row("CUDNN_ATTR_POINTWISE_ELU_ALPHA",
                                  CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_POINTWISE_MATH_PREC:
        return AttributeInfo::row("CUDNN_ATTR_POINTWISE_MATH_PREC",
                                  CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_DATA_TYPE);
    case CUDNN_ATTR_POINTWISE_MODE:
        return AttributeInfo::row("CUDNN_ATTR_POINTWISE_MODE",
                                  CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_POINTWISE_MODE);
    case CUDNN_ATTR_POINTWISE_NAN_PROPAGATION:
        return AttributeInfo::row("CUDNN_ATTR_POINTWISE_NAN_PROPAGATION",
                                  CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_NAN_PROPOGATION);
    case CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP:
        return AttributeInfo::row("CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP",
                                  CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
        return AttributeInfo::row("CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE",
                                  CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP:
        return AttributeInfo::row("CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP",
                                  CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA:
        return AttributeInfo::row("CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA",
                                  CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_POINTWISE_SWISH_BETA:
        return AttributeInfo::row("CUDNN_ATTR_POINTWISE_SWISH_BETA",
                                  CUDNN_BACKEND_POINTWISE_DESCRIPTOR,
                                  CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_REDUCTION_COMP_TYPE:
        return AttributeInfo::row("CUDNN_ATTR_REDUCTION_COMP_TYPE",
                                  CUDNN_BACKEND_REDUCTION_DESCRIPTOR,
                                  CUDNN_TYPE_DATA_TYPE);
    case CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC:
        return AttributeInfo::row("CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC",
                                  CUDNN_BACKEND_REDUCTION_DESCRIPTOR,
                                  CUDNN_TYPE_BOOLEAN);
    case CUDNN_ATTR_REDUCTION_OPERATOR:
        return AttributeInfo::row("CUDNN_ATTR_REDUCTION_OPERATOR",
                                  CUDNN_BACKEND_REDUCTION_DESCRIPTOR,
                                  CUDNN_TYPE_REDUCTION_OPERATOR_TYPE);
    case CUDNN_ATTR_RESAMPLE_COMP_TYPE:
        return AttributeInfo::row("CUDNN_ATTR_RESAMPLE_COMP_TYPE",
                                  CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                  CUDNN_TYPE_DATA_TYPE);
    case CUDNN_ATTR_RESAMPLE_MODE:
        return AttributeInfo::row("CUDNN_ATTR_RESAMPLE_MODE",
                                  CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                  CUDNN_TYPE_RESAMPLE_MODE);
    case CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION:
        return AttributeInfo::row("CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION",
                                  CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                  CUDNN_TYPE_NAN_PROPOGATION);
    case CUDNN_ATTR_RESAMPLE_PADDING_MODE:
        return AttributeInfo::row("CUDNN_ATTR_RESAMPLE_PADDING_MODE",
                                  CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                  CUDNN_TYPE_PADDING_MODE);
    case CUDNN_ATTR_RESAMPLE_POST_PADDINGS:
        return AttributeInfo::row("CUDNN_ATTR_RESAMPLE_POST_PADDINGS",
                                  CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                  CUDNN_TYPE_FRACTION)
            .asArray(CUDNN_DIM_MAX);
    case CUDNN_ATTR_RESAMPLE_PRE_PADDINGS:
        return AttributeInfo::row("CUDNN_ATTR_RESAMPLE_PRE_PADDINGS",
                                  CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                  CUDNN_TYPE_FRACTION)
            .asArray(CUDNN_DIM_MAX);
    case CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS:
        return AttributeInfo::row("CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS",
                                  CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                  CUDNN_TYPE_INT64);
    case CUDNN_ATTR_RESAMPLE_STRIDES:
        return AttributeInfo::row("CUDNN_ATTR_RESAMPLE_STRIDES",
                                  CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                  CUDNN_TYPE_FRACTION)
            .asArray(CUDNN_DIM_MAX);
    case CUDNN_ATTR_RESAMPLE_WINDOW_DIMS:
        return AttributeInfo::row("CUDNN_ATTR_RESAMPLE_WINDOW_DIMS",
                                  CUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
                                  CUDNN_TYPE_FRACTION)
            .asArray(CUDNN_DIM_MAX);
    case CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY:
        return AttributeInfo::row("CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY",
                                  CUDNN_BACKEND_RNG_DESCRIPTOR,
                                  CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_RNG_DISTRIBUTION:
        return AttributeInfo::row("CUDNN_ATTR_RNG_DISTRIBUTION",
                                  CUDNN_BACKEND_RNG_DESCRIPTOR,
                                  CUDNN_TYPE_RNG_DISTRIBUTION);
    case CUDNN_ATTR_RNG_NORMAL_DIST_MEAN:
        return AttributeInfo::row(
            "CUDNN_ATTR_RNG_NORMAL_DIST_MEAN", CUDNN_BACKEND_RNG_DESCRIPTOR, CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION:
        return AttributeInfo::row("CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION",
                                  CUDNN_BACKEND_RNG_DESCRIPTOR,
                                  CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM:
        return AttributeInfo::row(
            "CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM", CUDNN_BACKEND_RNG_DESCRIPTOR, CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM:
        return AttributeInfo::row(
            "CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM", CUDNN_BACKEND_RNG_DESCRIPTOR, CUDNN_TYPE_DOUBLE);
    case CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT:
        return AttributeInfo::row(
            "CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT", CUDNN_BACKEND_TENSOR_DESCRIPTOR, CUDNN_TYPE_INT64);
    case CUDNN_ATTR_TENSOR_CONSTANT_VALUE:
        return AttributeInfo::row("CUDNN_ATTR_TENSOR_CONSTANT_VALUE",
                                  CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                                  CUDNN_TYPE_VOID_PTR);
    case CUDNN_ATTR_TENSOR_DATA_TYPE:
        return AttributeInfo::row(
            "CUDNN_ATTR_TENSOR_DATA_TYPE", CUDNN_BACKEND_TENSOR_DESCRIPTOR, CUDNN_TYPE_DATA_TYPE);
    case CUDNN_ATTR_TENSOR_DIMENSIONS:
        return AttributeInfo::row("CUDNN_ATTR_TENSOR_DIMENSIONS",
                                  CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .asArray(CUDNN_DIM_MAX + 1);
    case CUDNN_ATTR_TENSOR_IS_BY_VALUE:
        return AttributeInfo::row("CUDNN_ATTR_TENSOR_IS_BY_VALUE",
                                  CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                                  CUDNN_TYPE_BOOLEAN)
            .withDefault(0);
    case CUDNN_ATTR_TENSOR_IS_VIRTUAL:
        return AttributeInfo::row("CUDNN_ATTR_TENSOR_IS_VIRTUAL",
                                  CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                                  CUDNN_TYPE_BOOLEAN)
            .withDefault(0);
    case CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC:
        return AttributeInfo::row("CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC",
                                  CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                                  CUDNN_TYPE_BACKEND_DESCRIPTOR)
            .refersTo(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    case CUDNN_ATTR_TENSOR_REORDERING_MODE:
        return AttributeInfo::row("CUDNN_ATTR_TENSOR_REORDERING_MODE",
                                  CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                                  CUDNN_TYPE_TENSOR_REORDERING_MODE)
            .withDefault(CUDNN_TENSOR_REORDERING_NONE);
    case CUDNN_ATTR_TENSOR_STRIDES:
        return AttributeInfo::row(
                   "CUDNN_ATTR_TENSOR_STRIDES", CUDNN_BACKEND_TENSOR_DESCRIPTOR, CUDNN_TYPE_INT64)
            .asArray(CUDNN_DIM_MAX + 1);
    case CUDNN_ATTR_TENSOR_UNIQUE_ID:
        return AttributeInfo::row(
            "CUDNN_ATTR_TENSOR_UNIQUE_ID", CUDNN_BACKEND_TENSOR_DESCRIPTOR, CUDNN_TYPE_INT64);
    case CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION:
        return AttributeInfo::row("CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION",
                                  CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .withDefault(-1);
    case CUDNN_ATTR_TENSOR_VECTOR_COUNT:
        return AttributeInfo::row("CUDNN_ATTR_TENSOR_VECTOR_COUNT",
                                  CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .withDefault(1);
    case CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
        return AttributeInfo::row("CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS",
                                  CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                  CUDNN_TYPE_VOID_PTR)
            .asArray(UNBOUNDED_COUNT);
    case CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        return AttributeInfo::row("CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS",
                                  CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                  CUDNN_TYPE_INT64)
            .asArray(UNBOUNDED_COUNT);
    case CUDNN_ATTR_VARIANT_PACK_WORKSPACE:
        return AttributeInfo::row("CUDNN_ATTR_VARIANT_PACK_WORKSPACE",
                                  CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                                  CUDNN_TYPE_VOID_PTR);
    default:
        return std::nullopt;
    }
}

// Attribute name for messages; unknown values are reported by number.
inline std::string attributeName(cudnnBackendAttributeName_t name)
{
    const std::optional<AttributeInfo> info = attributeInfo(name);
    return info ? std::string(info->name) : "attribute " + std::to_string(static_cast<int>(name));
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend

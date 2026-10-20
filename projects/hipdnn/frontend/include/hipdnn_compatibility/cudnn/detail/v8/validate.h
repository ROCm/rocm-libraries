// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_PointWiseDesc.h, include/cudnn_frontend_Operation.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/type_mapping.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>

// Record factories: every validation and normalization rule of the cuDNN v8
// tensor, parameter and operation descriptors, including the ones the C API
// applies at SET time (enum range, array caps, zero denominators), so builders
// that call the core directly get them too. Inputs are plain values (unset is
// std::nullopt, a null pointer or false); labels name each input in messages
// and are supplied by the caller. Each factory runs its checks in HEAD's
// finalize order, so the first failing rule decides the status.
namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

template <typename T>
T requireValue(const std::optional<T>& value, std::string_view label)
{
    if(!value)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(label) + " is required");
    }
    return *value;
}

template <typename T>
std::vector<T> requireNonEmpty(std::vector<T> values, std::string_view label)
{
    if(values.empty())
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(label) + " is required (non-empty)");
    }
    return values;
}

template <typename T>
void requireRecord(const std::shared_ptr<const T>& record, std::string_view label)
{
    if(!record)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(label) + " is required");
    }
}

inline TensorPtr requireTensor(const TensorPtr& tensor, std::string_view label)
{
    requireRecord(tensor, label);
    return tensor;
}

// Same rule and text as the C API's SET-time count check.
template <typename T>
void requireMaxCount(const std::vector<T>& values, std::size_t maxCount, std::string_view label)
{
    if(values.size() > maxCount)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(label) + ": element count " + std::to_string(values.size())
                            + " outside [0, " + std::to_string(maxCount) + "]");
    }
}

// Last declared enumerator; cuDNN's C enums are dense from 0.
constexpr int lastEnumerator(cudnnDataType_t /*value*/) noexcept
{
    return CUDNN_DATA_UINT8x4;
}

constexpr int lastEnumerator(cudnnConvolutionMode_t /*value*/) noexcept
{
    return CUDNN_CROSS_CORRELATION;
}

constexpr int lastEnumerator(cudnnPointwiseMode_t /*value*/) noexcept
{
    return CUDNN_POINTWISE_TANH_FWD;
}

constexpr int lastEnumerator(cudnnReduceTensorOp_t /*value*/) noexcept
{
    return CUDNN_REDUCE_TENSOR_NORM2;
}

constexpr int lastEnumerator(cudnnResampleMode_t /*value*/) noexcept
{
    return CUDNN_RESAMPLE_NEAREST;
}

constexpr int lastEnumerator(cudnnPaddingMode_t /*value*/) noexcept
{
    return CUDNN_ZERO_PAD;
}

constexpr int lastEnumerator(cudnnRngDistribution_t /*value*/) noexcept
{
    return CUDNN_RNG_DISTRIBUTION_UNIFORM;
}

constexpr int lastEnumerator(cudnnBackendNormMode_t /*value*/) noexcept
{
    return CUDNN_RMS_NORM;
}

constexpr int lastEnumerator(cudnnBackendNormFwdPhase_t /*value*/) noexcept
{
    return CUDNN_NORM_FWD_TRAINING;
}

constexpr int lastEnumerator(cudnnBackendTensorReordering_t /*value*/) noexcept
{
    return CUDNN_TENSOR_REORDERING_F8_128x4;
}

// A value outside the declared enumerators is a caller error (BAD_PARAM, as
// at SET), not an unsupported value.
template <typename CEnum>
CEnum requireEnumerator(CEnum value, std::string_view label)
{
    const auto raw = static_cast<std::int64_t>(value);
    if(raw < 0 || raw > lastEnumerator(value))
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(label) + " = " + std::to_string(raw)
                            + " is not a valid enumerator");
    }
    return value;
}

template <typename CEnum>
[[noreturn]] void throwUnsupportedValue(CEnum value, std::string_view label)
{
    throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                    std::string(label) + " value " + std::to_string(static_cast<int>(value))
                        + " is not supported by the hipDNN cuDNN shim");
}

// cuDNN value -> shim FE enum, for values that are kept but never lowered.
template <typename CEnum>
auto requireFe(CEnum value, std::string_view label)
{
    const auto mapped = toFe(requireEnumerator(value, label));
    if(!mapped)
    {
        throwUnsupportedValue(value, label);
    }
    return *mapped;
}

// cuDNN value -> shim FE enum that hipDNN can also represent; anything else is
// NOT_SUPPORTED.
template <typename CEnum>
auto requireLowerable(CEnum value, std::string_view label)
{
    const auto mapped = requireFe(value, label);
    if(!hasHipdnnCounterpart(mapped))
    {
        throwUnsupportedValue(value, label);
    }
    return mapped;
}

inline std::int64_t requireSpatialDims(std::optional<std::int64_t> value, std::string_view label)
{
    const std::int64_t spatialDims = requireValue(value, label);
    if(spatialDims < 1)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(label) + " must be >= 1");
    }
    return spatialDims;
}

// The count must match the spatial dimension count and stay within the C API's
// CUDNN_DIM_MAX array cap.
template <typename T>
void requireSpatialCount(const std::vector<T>& values,
                         std::int64_t spatialDims,
                         std::string_view label)
{
    if(static_cast<std::int64_t>(values.size()) != spatialDims)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(label) + " has " + std::to_string(values.size())
                            + " elements, expected the spatial dimension count "
                            + std::to_string(spatialDims));
    }
    requireMaxCount(values, std::size_t{CUDNN_DIM_MAX}, label);
}

// Alpha/beta: absent means the default; any other value cannot be lowered.
inline void requireScaling(std::optional<double> value, double expected, std::string_view label)
{
    if(value && *value != expected)
    {
        // Default stream formatting matches %g: "0.5", "1", "nan".
        std::ostringstream message;
        message << label << " = " << *value << " is not supported by the hipDNN cuDNN shim; only "
                << expected << " is";
        throw ShimError(CUDNN_STATUS_NOT_SUPPORTED, message.str());
    }
}

inline void rejectUnlowerable(bool present, std::string_view label, std::string_view context)
{
    if(present)
    {
        throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                        std::string(label) + " is not supported by the hipDNN cuDNN shim"
                            + std::string(context));
    }
}

struct OptionalSlot
{
    std::string_view label;
    bool present;
};

// All of `slots` or none of them; the message names the first missing one.
inline void requireAllOrNone(std::initializer_list<OptionalSlot> slots)
{
    const OptionalSlot* present = nullptr;
    const OptionalSlot* missing = nullptr;
    for(const OptionalSlot& slot : slots)
    {
        if(slot.present)
        {
            present = present != nullptr ? present : &slot;
        }
        else
        {
            missing = missing != nullptr ? missing : &slot;
        }
    }
    if(present != nullptr && missing != nullptr)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(missing->label) + " is required when "
                            + std::string(present->label) + " is set");
    }
}

// ---- Tensor ----

struct TensorLabels
{
    std::string_view dataType;
    std::string_view dimensions;
    std::string_view strides;
    std::string_view uniqueId;
    std::string_view alignment;
    std::string_view vectorCount;
    std::string_view reorderingMode;
    std::string_view constantValue;
};

struct TensorInputs
{
    std::optional<cudnnDataType_t> dataType;
    std::vector<std::int64_t> dimensions;
    std::vector<std::int64_t> strides;
    std::optional<std::int64_t> uniqueId;
    std::optional<std::int64_t> alignment;
    std::int64_t vectorCount = 1;
    cudnnBackendTensorReordering_t reorderingMode = CUDNN_TENSOR_REORDERING_NONE;
    bool isVirtual = false;
    bool isByValue = false;
    // Set when a constant was supplied, even as a null pointer (rejected). The
    // bytes are copied: the caller's storage need not outlive the record.
    std::optional<const void*> constantValue;
    TensorPtr raggedOffset;
};

inline std::shared_ptr<const TensorRecord> makeTensorRecord(TensorInputs inputs,
                                                            const TensorLabels& labels)
{
    const cudnnDataType_t rawType = requireValue(inputs.dataType, labels.dataType);
    std::vector<std::int64_t> dimensions
        = requireNonEmpty(std::move(inputs.dimensions), labels.dimensions);
    std::vector<std::int64_t> strides = requireNonEmpty(std::move(inputs.strides), labels.strides);
    requireMaxCount(dimensions, std::size_t{CUDNN_DIM_MAX} + 1, labels.dimensions);
    requireMaxCount(strides, std::size_t{CUDNN_DIM_MAX} + 1, labels.strides);
    if(dimensions.size() != strides.size())
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(labels.dimensions) + " has " + std::to_string(dimensions.size())
                            + " elements but " + std::string(labels.strides) + " has "
                            + std::to_string(strides.size()));
    }
    for(std::size_t i = 0; i < dimensions.size(); ++i)
    {
        if(dimensions[i] <= 0 || strides[i] < 0)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "tensor dimension " + std::to_string(i) + " is "
                                + std::to_string(dimensions[i]) + " with stride "
                                + std::to_string(strides[i])
                                + "; dimensions must be positive and strides non-negative");
        }
    }
    const std::int64_t uniqueId = requireValue(inputs.uniqueId, labels.uniqueId);
    if(uniqueId < 0)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(labels.uniqueId) + " must be >= 0");
    }
    const std::int64_t alignment = requireValue(inputs.alignment, labels.alignment);
    if(alignment <= 0)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(labels.alignment) + " must be > 0");
    }

    const DataType_t dataType = requireLowerable(rawType, labels.dataType);
    if(inputs.vectorCount != 1)
    {
        throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                        std::string(labels.vectorCount) + " other than 1 is not supported");
    }
    if(requireEnumerator(inputs.reorderingMode, labels.reorderingMode)
       != CUDNN_TENSOR_REORDERING_NONE)
    {
        throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                        std::string(labels.reorderingMode) + " other than NONE is not supported");
    }

    std::optional<std::vector<std::byte>> constantValue;
    if(inputs.constantValue)
    {
        const void* source = *inputs.constantValue;
        if(source == nullptr)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(labels.constantValue) + " is null");
        }
        // The element size is rounded up to whole bytes: upstream stores a
        // bool constant as one byte.
        const std::size_t size = (get_element_size_in_bits(dataType) + 7) / 8;
        if(size == 0)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            std::string(labels.constantValue)
                                + ": the tensor data type has no size");
        }
        constantValue.emplace(size);
        std::memcpy(constantValue->data(), source, size);
    }

    return std::make_shared<const TensorRecord>(
        TensorRecord::Fields{dataType,
                             std::move(dimensions),
                             std::move(strides),
                             uniqueId,
                             alignment,
                             inputs.isVirtual,
                             inputs.isByValue,
                             std::move(constantValue),
                             std::move(inputs.raggedOffset)});
}

// ---- Parameter descriptors ----

struct ConvolutionLabels
{
    std::string_view computeType;
    std::string_view mode;
    std::string_view spatialDims;
    std::string_view prePaddings;
    std::string_view postPaddings;
    std::string_view dilations;
    std::string_view filterStrides;
};

struct ConvolutionInputs
{
    std::optional<cudnnDataType_t> computeType;
    std::optional<cudnnConvolutionMode_t> mode;
    std::optional<std::int64_t> spatialDims;
    std::vector<std::int64_t> prePaddings;
    std::vector<std::int64_t> postPaddings;
    std::vector<std::int64_t> dilations;
    std::vector<std::int64_t> filterStrides;
};

inline std::shared_ptr<const ConvolutionRecord> makeConvRecord(ConvolutionInputs inputs,
                                                               const ConvolutionLabels& labels)
{
    const cudnnDataType_t rawComputeType = requireValue(inputs.computeType, labels.computeType);
    const cudnnConvolutionMode_t rawMode = requireValue(inputs.mode, labels.mode);
    const std::int64_t spatialDims = requireSpatialDims(inputs.spatialDims, labels.spatialDims);
    requireSpatialCount(inputs.prePaddings, spatialDims, labels.prePaddings);
    requireSpatialCount(inputs.postPaddings, spatialDims, labels.postPaddings);
    requireSpatialCount(inputs.dilations, spatialDims, labels.dilations);
    requireSpatialCount(inputs.filterStrides, spatialDims, labels.filterStrides);
    if(inputs.filterStrides.front() <= 0)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(labels.filterStrides) + "[0] must be > 0");
    }

    const DataType_t computeType = requireLowerable(rawComputeType, labels.computeType);
    const ConvolutionMode_t mode = requireLowerable(rawMode, labels.mode);
    return std::make_shared<const ConvolutionRecord>(
        ConvolutionRecord::Fields{computeType,
                                  mode,
                                  spatialDims,
                                  std::move(inputs.prePaddings),
                                  std::move(inputs.postPaddings),
                                  std::move(inputs.dilations),
                                  std::move(inputs.filterStrides)});
}

struct PointwiseLabels
{
    std::string_view mode;
    std::string_view computeType;
    std::string_view reluLowerClipSlope;
    std::string_view eluAlpha;
    std::string_view softplusBeta;
    std::string_view swishBeta;
};

// The mode parameters as set; makePointwiseRecord normalizes them.
struct PointwiseInputs
{
    std::optional<cudnnPointwiseMode_t> mode;
    std::optional<cudnnDataType_t> computeType;
    std::optional<double> reluLowerClip;
    std::optional<double> reluUpperClip;
    std::optional<double> reluLowerClipSlope;
    std::optional<double> eluAlpha;
    std::optional<double> softplusBeta;
    std::optional<double> swishBeta;
    std::optional<std::int64_t> axis;
};

// Keeps only the mode parameters that differ from upstream's defaults, plus the
// ReLU lower clip unless a slope is set. Upstream sets every parameter of the
// mode, defaults included, and MIOpen rejects a fusion that carries the others
// at all.
inline std::shared_ptr<const PointwiseRecord> makePointwiseRecord(const PointwiseInputs& inputs,
                                                                  const PointwiseLabels& labels)
{
    const cudnnPointwiseMode_t rawMode = requireValue(inputs.mode, labels.mode);
    const cudnnDataType_t rawComputeType = requireValue(inputs.computeType, labels.computeType);
    for(const auto& [value, label] :
        {std::pair{inputs.reluLowerClipSlope, labels.reluLowerClipSlope},
         std::pair{inputs.eluAlpha, labels.eluAlpha},
         std::pair{inputs.softplusBeta, labels.softplusBeta},
         std::pair{inputs.swishBeta, labels.swishBeta}})
    {
        if(value && !std::isfinite(*value))
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, std::string(label) + " must be finite");
        }
    }

    const auto nonDefault = [](std::optional<double> value, double defaultValue) {
        return value && *value != defaultValue ? value : std::nullopt;
    };
    PointwiseRecord::Fields fields;
    fields.mode = requireLowerable(rawMode, labels.mode);
    fields.computeType = requireLowerable(rawComputeType, labels.computeType);
    // Defaults from PointWiseDesc_v8 (cudnn_frontend_PointWiseDesc.h).
    fields.reluLowerClipSlope = nonDefault(inputs.reluLowerClipSlope, 0.0);
    fields.eluAlpha = nonDefault(inputs.eluAlpha, 1.0);
    fields.softplusBeta = nonDefault(inputs.softplusBeta, 1.0);
    fields.swishBeta = nonDefault(inputs.swishBeta, 1.0);
    // Every engine reads a lower clip of 0 as standard ReLU, and hipBLASLt's ReLU
    // epilogue requires it present. Beside a slope it stays elided: hipBLASLt
    // would take {0, slope} for plain ReLU and drop the slope.
    fields.reluLowerClip
        = fields.reluLowerClipSlope ? nonDefault(inputs.reluLowerClip, 0.0) : inputs.reluLowerClip;
    // The upper-clip default is DBL_MAX, which upstream clamps to FLT_MAX for
    // FLOAT math; any bound at or above FLT_MAX clips nothing once narrowed.
    fields.reluUpperClip
        = inputs.reluUpperClip
                  && *inputs.reluUpperClip < static_cast<double>(std::numeric_limits<float>::max())
              ? inputs.reluUpperClip
              : std::nullopt;
    fields.axis = inputs.axis && *inputs.axis != -1 ? inputs.axis : std::nullopt;
    return std::make_shared<const PointwiseRecord>(fields);
}

struct MatmulLabels
{
    std::string_view computeType;
    std::string_view paddingValue;
};

struct MatmulInputs
{
    std::optional<cudnnDataType_t> computeType;
    bool hasPaddingValue = false;
};

inline std::shared_ptr<const MatmulRecord> makeMatmulRecord(const MatmulInputs& inputs,
                                                            const MatmulLabels& labels)
{
    const cudnnDataType_t rawComputeType = requireValue(inputs.computeType, labels.computeType);
    if(inputs.hasPaddingValue)
    {
        throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                        std::string(labels.paddingValue) + " is not supported");
    }
    return std::make_shared<const MatmulRecord>(
        requireLowerable(rawComputeType, labels.computeType));
}

struct ReductionLabels
{
    std::string_view mode;
    std::string_view computeType;
};

struct ReductionInputs
{
    std::optional<cudnnReduceTensorOp_t> mode;
    std::optional<cudnnDataType_t> computeType;
    bool isDeterministic = false;
};

inline std::shared_ptr<const ReductionRecord> makeReductionRecord(const ReductionInputs& inputs,
                                                                  const ReductionLabels& labels)
{
    const cudnnReduceTensorOp_t rawMode = requireValue(inputs.mode, labels.mode);
    const cudnnDataType_t rawComputeType = requireValue(inputs.computeType, labels.computeType);
    const ReductionMode_t mode = requireLowerable(rawMode, labels.mode);
    const DataType_t computeType = requireLowerable(rawComputeType, labels.computeType);
    return std::make_shared<const ReductionRecord>(
        ReductionRecord::Fields{mode, computeType, inputs.isDeterministic});
}

struct ResampleLabels
{
    std::string_view mode;
    std::string_view computeType;
    std::string_view paddingMode;
    std::string_view spatialDims;
    std::string_view windowDims;
    std::string_view prePaddings;
    std::string_view postPaddings;
    std::string_view strides;
};

struct ResampleInputs
{
    std::optional<cudnnResampleMode_t> mode;
    std::optional<cudnnDataType_t> computeType;
    std::optional<cudnnPaddingMode_t> paddingMode;
    std::optional<std::int64_t> spatialDims;
    std::vector<cudnnFraction_t> windowDims;
    std::vector<cudnnFraction_t> prePaddings;
    std::vector<cudnnFraction_t> postPaddings;
    std::vector<cudnnFraction_t> strides;
};

inline std::vector<std::int64_t> integralFractions(const std::vector<cudnnFraction_t>& fractions,
                                                   std::string_view label)
{
    std::vector<std::int64_t> values;
    values.reserve(fractions.size());
    for(std::size_t i = 0; i < fractions.size(); ++i)
    {
        const cudnnFraction_t& fraction = fractions[i];
        if(fraction.denominator == 0)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(label) + "[" + std::to_string(i)
                                + "] has a zero denominator");
        }
        // INT64_MIN / -1 overflows; its quotient is unrepresentable anyway.
        const bool overflows = fraction.denominator == -1
                               && fraction.numerator == std::numeric_limits<std::int64_t>::min();
        if(overflows || fraction.numerator % fraction.denominator != 0)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            std::string(label) + "[" + std::to_string(i)
                                + "] = " + std::to_string(fraction.numerator) + "/"
                                + std::to_string(fraction.denominator)
                                + " is not an integer; fractional values are not supported");
        }
        values.push_back(fraction.numerator / fraction.denominator);
    }
    return values;
}

inline std::shared_ptr<const ResampleRecord> makeResampleRecord(const ResampleInputs& inputs,
                                                                const ResampleLabels& labels)
{
    const cudnnResampleMode_t rawMode = requireValue(inputs.mode, labels.mode);
    const cudnnDataType_t rawComputeType = requireValue(inputs.computeType, labels.computeType);
    // Upstream's builder refuses to build without a padding mode, and hipDNN's
    // resample descriptors require one.
    const cudnnPaddingMode_t rawPadding = requireValue(inputs.paddingMode, labels.paddingMode);
    // Upstream sends 0 when the builder never set it.
    const std::int64_t spatialDims = requireSpatialDims(inputs.spatialDims, labels.spatialDims);
    requireSpatialCount(inputs.windowDims, spatialDims, labels.windowDims);
    requireSpatialCount(inputs.prePaddings, spatialDims, labels.prePaddings);
    requireSpatialCount(inputs.postPaddings, spatialDims, labels.postPaddings);
    requireSpatialCount(inputs.strides, spatialDims, labels.strides);

    ResampleRecord::Fields fields;
    // NEAREST and BILINEAR have no hipDNN resample mode.
    fields.mode = requireLowerable(rawMode, labels.mode);
    fields.computeType = requireLowerable(rawComputeType, labels.computeType);
    fields.paddingMode = requireLowerable(rawPadding, labels.paddingMode);
    fields.windowDims = integralFractions(inputs.windowDims, labels.windowDims);
    fields.prePaddings = integralFractions(inputs.prePaddings, labels.prePaddings);
    fields.postPaddings = integralFractions(inputs.postPaddings, labels.postPaddings);
    fields.strides = integralFractions(inputs.strides, labels.strides);
    return std::make_shared<const ResampleRecord>(std::move(fields));
}

inline std::shared_ptr<const RngRecord>
    makeRngRecord(std::optional<cudnnRngDistribution_t> distribution, std::string_view label)
{
    return std::make_shared<const RngRecord>(requireFe(requireValue(distribution, label), label));
}

// ---- Operations ----

enum class NormFamily : std::uint8_t
{
    BATCH,
    LAYER,
    RMS
};

inline NormFamily requireNormFamily(cudnnBackendNormMode_t mode, std::string_view label)
{
    const char* unsupported = "an unknown mode";
    switch(requireEnumerator(mode, label))
    {
    case CUDNN_BATCH_NORM:
        return NormFamily::BATCH;
    case CUDNN_LAYER_NORM:
        return NormFamily::LAYER;
    case CUDNN_RMS_NORM:
        return NormFamily::RMS;
    case CUDNN_ADA_LAYER_NORM:
        unsupported = "CUDNN_ADA_LAYER_NORM";
        break;
    case CUDNN_GROUP_NORM:
        unsupported = "CUDNN_GROUP_NORM";
        break;
    case CUDNN_INSTANCE_NORM:
        unsupported = "CUDNN_INSTANCE_NORM";
        break;
    default:
        break;
    }
    throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                    std::string(label) + " " + unsupported
                        + " is not supported by the hipDNN cuDNN shim");
}

// hipDNN's layernorm normalized-dim count, derived from the scale shape exactly
// as LayerNormNode::infer_properties_node does: the trailing non-1 scale dims of
// a full-rank scale (at least 1), else every scale dim.
inline std::int64_t layernormNormalizedDimCount(const TensorRecord& x, const TensorRecord& scale)
{
    const std::vector<std::int64_t>& scaleDims = scale.dimensions();
    const auto rank = static_cast<std::int64_t>(scaleDims.size());
    if(x.dimensions().size() != scaleDims.size())
    {
        return rank;
    }
    std::int64_t count = 1;
    for(std::int64_t i = rank - 1; i >= 0; --i)
    {
        if(scaleDims[static_cast<std::size_t>(i)] == 1)
        {
            break;
        }
        count = rank - i;
    }
    return count;
}

// The three convolution operations differ only in their labels and which
// tensor is the output; `tensors` is in the record's slot order (X, W, Y /
// DY, W, DX / DY, X, DW).
struct ConvolutionOperationLabels
{
    std::string_view convolution;
    std::string_view alpha;
    std::string_view beta;
    std::array<std::string_view, 3> tensors;
    std::string_view spatialDims;
};

struct ConvolutionOperationInputs
{
    std::shared_ptr<const ConvolutionRecord> convolution;
    std::array<TensorPtr, 3> tensors;
    std::optional<double> alpha;
    std::optional<double> beta;
};

template <typename OperationRecordT>
std::shared_ptr<const OperationRecordT>
    makeOperationRecord(const ConvolutionOperationInputs& inputs,
                        const ConvolutionOperationLabels& labels)
{
    static_assert(std::is_same_v<OperationRecordT, ConvolutionForwardOperationRecord>
                  || std::is_same_v<OperationRecordT, ConvolutionBackwardDataOperationRecord>
                  || std::is_same_v<OperationRecordT, ConvolutionBackwardFilterOperationRecord>);
    requireRecord(inputs.convolution, labels.convolution);
    for(std::size_t i = 0; i < inputs.tensors.size(); ++i)
    {
        requireRecord(inputs.tensors[i], labels.tensors[i]);
    }
    const std::size_t rank = static_cast<std::size_t>(inputs.convolution->spatialDims()) + 2;
    for(std::size_t i = 0; i < inputs.tensors.size(); ++i)
    {
        const std::size_t actual = inputs.tensors[i]->dimensions().size();
        if(actual != rank)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(labels.tensors[i]) + " has rank " + std::to_string(actual)
                                + ", expected " + std::string(labels.spatialDims)
                                + " + 2 = " + std::to_string(rank));
        }
    }
    requireScaling(inputs.alpha, 1.0, labels.alpha);
    requireScaling(inputs.beta, 0.0, labels.beta);
    return std::make_shared<const OperationRecordT>(typename OperationRecordT::Fields{
        inputs.convolution, inputs.tensors[0], inputs.tensors[1], inputs.tensors[2]});
}

struct PointwiseOperationLabels
{
    std::string_view pointwise;
    std::string_view x;
    std::string_view b;
    std::string_view t;
    std::string_view y;
    std::string_view dy;
    std::string_view dx;
    std::string_view alpha1;
    std::string_view alpha2;
};

struct PointwiseOperationInputs
{
    std::shared_ptr<const PointwiseRecord> pointwise;
    TensorPtr x;
    TensorPtr b;
    TensorPtr t;
    TensorPtr y;
    TensorPtr dy;
    TensorPtr dx;
    std::optional<double> alpha1;
    std::optional<double> alpha2;
};

inline std::shared_ptr<const PointwiseOperationRecord>
    makeOperationRecord(const PointwiseOperationInputs& inputs,
                        const PointwiseOperationLabels& labels)
{
    requireRecord(inputs.pointwise, labels.pointwise);
    const std::optional<PointwiseModeTraits> traits = pointwiseModeTraits(inputs.pointwise->mode());
    if(!traits)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(labels.pointwise) + " has no pointwise mode");
    }
    const bool backward = traits->modeClass == PointwiseModeClass::ACTIVATION_BACKWARD;

    PointwiseOperationRecord::Fields fields;
    fields.pointwise = inputs.pointwise;
    fields.traits = *traits;
    requireRecord(inputs.x, labels.x);
    if(backward)
    {
        fields.in0 = requireTensor(inputs.dy, labels.dy);
        fields.out0 = requireTensor(inputs.dx, labels.dx);
        fields.in1 = inputs.x;
    }
    else
    {
        fields.out0 = requireTensor(inputs.y, labels.y);
        if(traits->portCount >= 3)
        {
            fields.in1 = requireTensor(inputs.b, labels.b);
        }
        if(traits->portCount >= 4)
        {
            fields.in2 = requireTensor(inputs.t, labels.t);
        }
        fields.in0 = inputs.x;
    }

    const auto rejectUnusedSlot = [&traits](const TensorPtr& tensor, std::string_view label) {
        if(tensor)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            std::string(label) + " is not used by a pointwise "
                                + pointwiseModeClassName(traits->modeClass) + " mode with "
                                + std::to_string(traits->portCount) + " ports");
        }
    };
    if(backward)
    {
        rejectUnusedSlot(inputs.y, labels.y);
        rejectUnusedSlot(inputs.b, labels.b);
        rejectUnusedSlot(inputs.t, labels.t);
    }
    else
    {
        rejectUnusedSlot(inputs.dy, labels.dy);
        rejectUnusedSlot(inputs.dx, labels.dx);
        // Upstream sends B only for port >= 3 and T only for port 4.
        if(traits->portCount < 3)
        {
            rejectUnusedSlot(inputs.b, labels.b);
        }
        if(traits->portCount < 4)
        {
            rejectUnusedSlot(inputs.t, labels.t);
        }
    }

    requireScaling(inputs.alpha1, 1.0, labels.alpha1);
    requireScaling(inputs.alpha2, 1.0, labels.alpha2);
    return std::make_shared<const PointwiseOperationRecord>(std::move(fields));
}

struct MatmulOperationLabels
{
    std::string_view matmul;
    std::string_view a;
    std::string_view b;
    std::string_view c;
    std::string_view mOverride;
    std::string_view nOverride;
    std::string_view kOverride;
};

struct MatmulOperationInputs
{
    std::shared_ptr<const MatmulRecord> matmul;
    TensorPtr a;
    TensorPtr b;
    TensorPtr c;
    bool hasMOverride = false;
    bool hasNOverride = false;
    bool hasKOverride = false;
};

inline std::shared_ptr<const MatmulOperationRecord>
    makeOperationRecord(const MatmulOperationInputs& inputs, const MatmulOperationLabels& labels)
{
    requireRecord(inputs.matmul, labels.matmul);
    requireRecord(inputs.a, labels.a);
    requireRecord(inputs.b, labels.b);
    requireRecord(inputs.c, labels.c);
    rejectUnlowerable(inputs.hasMOverride, labels.mOverride, "");
    rejectUnlowerable(inputs.hasNOverride, labels.nOverride, "");
    rejectUnlowerable(inputs.hasKOverride, labels.kOverride, "");
    return std::make_shared<const MatmulOperationRecord>(
        MatmulOperationRecord::Fields{inputs.matmul, inputs.a, inputs.b, inputs.c});
}

struct ReductionOperationLabels
{
    std::string_view reduction;
    std::string_view x;
    std::string_view y;
};

struct ReductionOperationInputs
{
    std::shared_ptr<const ReductionRecord> reduction;
    TensorPtr x;
    TensorPtr y;
};

inline std::shared_ptr<const ReductionOperationRecord>
    makeOperationRecord(const ReductionOperationInputs& inputs,
                        const ReductionOperationLabels& labels)
{
    requireRecord(inputs.reduction, labels.reduction);
    requireRecord(inputs.x, labels.x);
    requireRecord(inputs.y, labels.y);
    return std::make_shared<const ReductionOperationRecord>(
        ReductionOperationRecord::Fields{inputs.reduction, inputs.x, inputs.y});
}

struct ResampleForwardOperationLabels
{
    std::string_view resample;
    std::string_view x;
    std::string_view y;
    std::string_view alpha;
    std::string_view beta;
};

struct ResampleForwardOperationInputs
{
    std::shared_ptr<const ResampleRecord> resample;
    TensorPtr x;
    TensorPtr y;
    TensorPtr index;
    std::optional<double> alpha;
    std::optional<double> beta;
};

inline std::shared_ptr<const ResampleForwardOperationRecord>
    makeOperationRecord(const ResampleForwardOperationInputs& inputs,
                        const ResampleForwardOperationLabels& labels)
{
    requireRecord(inputs.resample, labels.resample);
    requireRecord(inputs.x, labels.x);
    requireRecord(inputs.y, labels.y);
    requireScaling(inputs.alpha, 1.0, labels.alpha);
    requireScaling(inputs.beta, 0.0, labels.beta);
    return std::make_shared<const ResampleForwardOperationRecord>(
        ResampleForwardOperationRecord::Fields{inputs.resample, inputs.x, inputs.y, inputs.index});
}

struct ResampleBackwardOperationLabels
{
    std::string_view resample;
    std::string_view dx;
    std::string_view dy;
    std::string_view index;
    std::string_view alpha;
    std::string_view beta;
    std::string_view x;
    std::string_view y;
};

// hipDNN's resample backward has no X or Y slot, so those are only flagged.
struct ResampleBackwardOperationInputs
{
    std::shared_ptr<const ResampleRecord> resample;
    TensorPtr dx;
    TensorPtr dy;
    TensorPtr index;
    std::optional<double> alpha;
    std::optional<double> beta;
    bool hasX = false;
    bool hasY = false;
};

inline std::shared_ptr<const ResampleBackwardOperationRecord>
    makeOperationRecord(const ResampleBackwardOperationInputs& inputs,
                        const ResampleBackwardOperationLabels& labels)
{
    requireRecord(inputs.resample, labels.resample);
    requireRecord(inputs.dx, labels.dx);
    requireRecord(inputs.dy, labels.dy);
    // hipDNN's max-pool backward needs the forward pass's argmax indices.
    if(inputs.resample->mode() == ResampleMode_t::MAXPOOL && !inputs.index)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(labels.index) + " is required for CUDNN_RESAMPLE_MAXPOOL");
    }
    requireScaling(inputs.alpha, 1.0, labels.alpha);
    requireScaling(inputs.beta, 0.0, labels.beta);
    rejectUnlowerable(inputs.hasX, labels.x, " for resample backward");
    rejectUnlowerable(inputs.hasY, labels.y, " for resample backward");
    return std::make_shared<const ResampleBackwardOperationRecord>(
        ResampleBackwardOperationRecord::Fields{
            inputs.resample, inputs.dx, inputs.dy, inputs.index});
}

struct NormForwardOperationLabels
{
    std::string_view mode;
    std::string_view phase;
    std::string_view x;
    std::string_view scale;
    std::string_view bias;
    std::string_view epsilon;
    std::string_view y;
    std::string_view mean;
    std::string_view invVariance;
    std::string_view inputRunningMean;
    std::string_view inputRunningVariance;
    std::string_view expAvgFactor;
    std::string_view outputRunningMean;
    std::string_view outputRunningVariance;
    std::string_view peerStats;
};

struct NormForwardOperationInputs
{
    std::optional<cudnnBackendNormMode_t> mode;
    std::optional<cudnnBackendNormFwdPhase_t> phase;
    TensorPtr x;
    TensorPtr scale;
    TensorPtr bias;
    TensorPtr epsilon;
    TensorPtr y;
    TensorPtr mean;
    TensorPtr invVariance;
    TensorPtr inputRunningMean;
    TensorPtr inputRunningVariance;
    TensorPtr expAvgFactor;
    TensorPtr outputRunningMean;
    TensorPtr outputRunningVariance;
    bool hasPeerStats = false;
};

inline std::shared_ptr<const NormForwardOperationRecord>
    makeOperationRecord(const NormForwardOperationInputs& inputs,
                        const NormForwardOperationLabels& labels)
{
    const NormFamily family
        = requireNormFamily(requireValue(inputs.mode, labels.mode), labels.mode);
    const cudnnBackendNormFwdPhase_t rawPhase = requireValue(inputs.phase, labels.phase);
    NormForwardOperationRecord::Fields fields;
    fields.phase = requireFe(rawPhase, labels.phase);
    OperationTarget target = OperationTarget::RMSNORM;
    if(family == NormFamily::BATCH)
    {
        target = rawPhase == CUDNN_NORM_FWD_TRAINING ? OperationTarget::BATCHNORM
                                                     : OperationTarget::BATCHNORM_INFERENCE;
    }
    else if(family == NormFamily::LAYER)
    {
        target = OperationTarget::LAYERNORM;
    }

    fields.x = requireTensor(inputs.x, labels.x);
    fields.scale = requireTensor(inputs.scale, labels.scale);
    if(target == OperationTarget::BATCHNORM_INFERENCE)
    {
        fields.mean = requireTensor(inputs.mean, labels.mean);
        fields.invVariance = requireTensor(inputs.invVariance, labels.invVariance);
    }
    if(target != OperationTarget::RMSNORM)
    {
        fields.bias = requireTensor(inputs.bias, labels.bias);
    }
    if(target != OperationTarget::BATCHNORM_INFERENCE)
    {
        fields.epsilon = requireTensor(inputs.epsilon, labels.epsilon);
    }
    fields.y = requireTensor(inputs.y, labels.y);
    if(target == OperationTarget::BATCHNORM || target == OperationTarget::LAYERNORM)
    {
        requireAllOrNone({{labels.mean, inputs.mean != nullptr},
                          {labels.invVariance, inputs.invVariance != nullptr}});
    }
    if(target == OperationTarget::BATCHNORM)
    {
        requireAllOrNone({{labels.inputRunningMean, inputs.inputRunningMean != nullptr},
                          {labels.inputRunningVariance, inputs.inputRunningVariance != nullptr},
                          {labels.expAvgFactor, inputs.expAvgFactor != nullptr},
                          {labels.outputRunningMean, inputs.outputRunningMean != nullptr},
                          {labels.outputRunningVariance, inputs.outputRunningVariance != nullptr}});
    }

    rejectUnlowerable(inputs.hasPeerStats, labels.peerStats, "");
    if(target != OperationTarget::BATCHNORM)
    {
        for(const auto& [tensor, label] :
            {std::pair{&inputs.inputRunningMean, labels.inputRunningMean},
             std::pair{&inputs.inputRunningVariance, labels.inputRunningVariance},
             std::pair{&inputs.expAvgFactor, labels.expAvgFactor},
             std::pair{&inputs.outputRunningMean, labels.outputRunningMean},
             std::pair{&inputs.outputRunningVariance, labels.outputRunningVariance}})
        {
            rejectUnlowerable(*tensor != nullptr, label, " for this normalization mode and phase");
        }
    }
    if(target == OperationTarget::RMSNORM)
    {
        rejectUnlowerable(inputs.mean != nullptr, labels.mean, " for CUDNN_RMS_NORM");
        fields.bias = inputs.bias;
    }
    if(target != OperationTarget::BATCHNORM_INFERENCE)
    {
        fields.mean = inputs.mean;
        fields.invVariance = inputs.invVariance;
    }
    if(target == OperationTarget::BATCHNORM)
    {
        fields.inputRunningMean = inputs.inputRunningMean;
        fields.inputRunningVariance = inputs.inputRunningVariance;
        fields.expAvgFactor = inputs.expAvgFactor;
        fields.outputRunningMean = inputs.outputRunningMean;
        fields.outputRunningVariance = inputs.outputRunningVariance;
    }
    if(target == OperationTarget::LAYERNORM)
    {
        fields.normalizedDimCount = layernormNormalizedDimCount(*fields.x, *fields.scale);
    }
    return std::make_shared<const NormForwardOperationRecord>(target, std::move(fields));
}

struct NormBackwardOperationLabels
{
    std::string_view mode;
    std::string_view x;
    std::string_view dy;
    std::string_view scale;
    std::string_view mean;
    std::string_view invVariance;
    std::string_view epsilon;
    std::string_view dx;
    std::string_view dscale;
    std::string_view dbias;
    std::string_view peerStats;
};

struct NormBackwardOperationInputs
{
    std::optional<cudnnBackendNormMode_t> mode;
    TensorPtr x;
    TensorPtr dy;
    TensorPtr scale;
    TensorPtr mean;
    TensorPtr invVariance;
    TensorPtr epsilon;
    TensorPtr dx;
    TensorPtr dscale;
    TensorPtr dbias;
    bool hasPeerStats = false;
};

inline std::shared_ptr<const NormBackwardOperationRecord>
    makeOperationRecord(const NormBackwardOperationInputs& inputs,
                        const NormBackwardOperationLabels& labels)
{
    const NormFamily family
        = requireNormFamily(requireValue(inputs.mode, labels.mode), labels.mode);
    OperationTarget target = OperationTarget::RMSNORM_BACKWARD;
    if(family == NormFamily::BATCH)
    {
        target = OperationTarget::BATCHNORM_BACKWARD;
    }
    else if(family == NormFamily::LAYER)
    {
        target = OperationTarget::LAYERNORM_BACKWARD;
    }

    NormBackwardOperationRecord::Fields fields;
    fields.x = requireTensor(inputs.x, labels.x);
    fields.dy = requireTensor(inputs.dy, labels.dy);
    fields.scale = requireTensor(inputs.scale, labels.scale);
    // Without saved statistics hipDNN's batchnorm backward would need the
    // epsilon, which it cannot take.
    if(target == OperationTarget::BATCHNORM_BACKWARD)
    {
        fields.mean = requireTensor(inputs.mean, labels.mean);
    }
    if(target != OperationTarget::LAYERNORM_BACKWARD)
    {
        fields.invVariance = requireTensor(inputs.invVariance, labels.invVariance);
    }
    fields.dx = requireTensor(inputs.dx, labels.dx);
    fields.dscale = requireTensor(inputs.dscale, labels.dscale);
    if(target != OperationTarget::RMSNORM_BACKWARD)
    {
        fields.dbias = requireTensor(inputs.dbias, labels.dbias);
    }
    if(target == OperationTarget::LAYERNORM_BACKWARD)
    {
        requireAllOrNone({{labels.mean, inputs.mean != nullptr},
                          {labels.invVariance, inputs.invVariance != nullptr}});
    }

    rejectUnlowerable(inputs.hasPeerStats, labels.peerStats, "");
    if(target == OperationTarget::RMSNORM_BACKWARD)
    {
        rejectUnlowerable(inputs.mean != nullptr, labels.mean, " for CUDNN_RMS_NORM");
        fields.dbias = inputs.dbias;
    }
    if(target == OperationTarget::LAYERNORM_BACKWARD)
    {
        fields.mean = inputs.mean;
        fields.invVariance = inputs.invVariance;
        fields.epsilon = inputs.epsilon;
        fields.normalizedDimCount = layernormNormalizedDimCount(*fields.x, *fields.scale);
    }
    return std::make_shared<const NormBackwardOperationRecord>(target, std::move(fields));
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
